import torch
import time


def print_hx_stats(pred, modality_order, epoch=None):
    hx = pred['hx']  # [num_sources+1, batch_size]
    num_sources = len(modality_order)

    epoch_str = f"Epoch {epoch:3d}" if epoch is not None else "Current"

    print(f"\n  {'='*60}")
    print(f"  {epoch_str} | hx Statistics:")
    print(f"  {'-'*60}")

    for i, name in enumerate(modality_order):
        hx_i = hx[i].detach().cpu().numpy()
        print(f"  {name:10s} | mean: {hx_i.mean():10.4f} | std: {hx_i.std():8.4f} | "
              f"min: {hx_i.min():8.4f} | max: {hx_i.max():8.4f}")

    # Combined (fused) hx
    hx_comb = hx[-1].detach().cpu().numpy()
    print(f"  {'Combined':10s} | mean: {hx_comb.mean():10.4f} | std: {hx_comb.std():8.4f} | "
          f"min: {hx_comb.min():8.4f} | max: {hx_comb.max():8.4f}")

    # hx ratio per modality (helps interpret the fusion weights)
    print(f"  {'-'*60}")
    print(f"  hx Ratios (contribution to fusion):")
    total_hx = sum(hx[i].detach().cpu().numpy().mean() for i in range(num_sources))
    for i, name in enumerate(modality_order):
        ratio = hx[i].detach().cpu().numpy().mean() / (total_hx + 1e-8) * 100
        print(f"  {name:10s} | ratio: {ratio:6.2f}%")
    print(f"  {'='*60}\n")


class EVREGTrainer:
    """EVREG model trainer"""

    def __init__(self, model, optimizer, scheduler, criterion, criterion_eval, config, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.criterion_eval = criterion_eval
        self.config = config
        self.device = device

        self.model.to(self.device)

    def _move_batch(self, inputs, log_t, events, masks):
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)

        log_t = log_t.to(self.device)
        events = events.to(self.device)

        if masks is not None:
            if isinstance(masks, dict):
                masks = {k: v.to(self.device) for k, v in masks.items()}
            else:
                masks = masks.to(self.device)

        return inputs, log_t, events, masks

    def train_epoch(self, train_loader, use_mask=False):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        skip_nan_logt = 0
        skip_nan_pred = 0
        skip_nan_loss = 0

        for batch_data in train_loader:
            if use_mask:
                inputs, log_t, events, masks = batch_data
            else:
                inputs, log_t, events = batch_data
                masks = None

            inputs, log_t, events, masks = self._move_batch(inputs, log_t, events, masks)

            # Forward pass
            pred = self.model(inputs, masks=masks) if use_mask else self.model(inputs)

            # Check for NaN
            if torch.isnan(log_t).any():
                print(f"  WARNING: NaN in batch durations, skipping...")
                skip_nan_logt += 1
                continue

            if torch.isnan(pred['mux']).any() or torch.isnan(pred['sig2x']).any():
                print(f"  WARNING: NaN in predictions, skipping...")
                skip_nan_pred += 1
                continue

            # Compute loss
            self.optimizer.zero_grad()
            loss = self.criterion(
                log_t,
                nu=self.config.loss.nu,
                events=events,
                xi=self.config.loss.xi,
                rho=self.config.loss.rho,
                lambd=self.config.loss.train_lambd,
                pred=pred,
                sigma=self.config.loss.sigma,
                c=self.config.loss.c
            )

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss, skipping...")
                skip_nan_loss += 1
                continue

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        skipped = skip_nan_logt + skip_nan_pred + skip_nan_loss
        if skipped > 0:
            print(
                f"  WARNING: Skipped {skipped} batches (log_t: {skip_nan_logt}, "
                f"pred: {skip_nan_pred}, loss: {skip_nan_loss})"
            )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(self, val_loader, use_mask=False):
        """Evaluate the loss on a loader (used to track test loss during training)"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if use_mask:
                    inputs, log_t, events, masks = batch_data
                else:
                    inputs, log_t, events = batch_data
                    masks = None

                inputs, log_t, events, masks = self._move_batch(inputs, log_t, events, masks)

                pred = self.model(inputs, masks=masks) if use_mask else self.model(inputs)

                loss = self.criterion_eval(
                    log_t,
                    nu=self.config.loss.nu,
                    events=events,
                    xi=self.config.loss.xi,
                    rho=self.config.loss.rho,
                    lambd=self.config.loss.train_lambd,
                    pred=pred,
                    sigma=self.config.loss.sigma,
                    c=self.config.loss.c
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train_no_val(self, train_loader, save_path, use_mask=False, print_every=50, max_epochs=None, test_loader=None, print_hx_every=None):
        epochs = max_epochs if max_epochs is not None else self.config.training.max_epochs

        print(f"\n{'=' * 80}")
        print(f"Training (NO validation, fixed {epochs} epochs)...")
        if test_loader is not None:
            print("  -> Tracking test loss at each epoch")
        if print_hx_every is not None:
            print(f"  -> Printing hx statistics every {print_hx_every} epochs")
        print(f"{'=' * 80}")

        start_time = time.time()
        best_train_loss = float('inf')
        best_epoch = 0

        # History tracking
        train_loss_history = []
        test_loss_history = [] if test_loader is not None else None
        hx_history = []  # track hx over epochs

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, use_mask=use_mask)
            train_loss_history.append(train_loss)

            # Evaluate on test set if provided
            if test_loader is not None:
                test_loss = self.validate(test_loader, use_mask=use_mask)
                test_loss_history.append(test_loss)

            # Step scheduler
            self.scheduler.step()

            # Track best training loss (for reference only)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch + 1

            # Print progress
            if (epoch + 1) % print_every == 0:
                if test_loader is not None:
                    print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Test: {test_loss:.4f}")
                else:
                    print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f}")

            # Print hx statistics
            if print_hx_every is not None and (epoch + 1) % print_hx_every == 0:
                self.model.eval()
                with torch.no_grad():
                    # Use the first batch to compute hx stats
                    for batch_data in train_loader:
                        if use_mask:
                            inputs, log_t, events, masks = batch_data
                        else:
                            inputs, log_t, events = batch_data
                            masks = None
                        inputs, log_t, events, masks = self._move_batch(inputs, log_t, events, masks)
                        pred = self.model(inputs, masks=masks) if use_mask else self.model(inputs)

                        modality_order = pred.get('modality_order', list(inputs.keys()))
                        print_hx_stats(pred, modality_order, epoch=epoch+1)

                        # Record mean hx into history
                        hx_record = {'epoch': epoch + 1}
                        for i, name in enumerate(modality_order):
                            hx_record[f'hx_{name}_mean'] = pred['hx'][i].detach().cpu().numpy().mean()
                        hx_record['hx_combined_mean'] = pred['hx'][-1].detach().cpu().numpy().mean()
                        hx_history.append(hx_record)
                        break  # only the first batch
                self.model.train()

        # Save the final model (not early stopped)
        torch.save(self.model.state_dict(), save_path)
        print(f"  -> Final model saved")

        elapsed = (time.time() - start_time) / 60
        print(f"\nTraining time: {elapsed:.2f} min")
        print(f"Final epoch: {epochs} (Best train loss at epoch {best_epoch}: {best_train_loss:.4f})")

        history = {
            'train_loss': train_loss_history,
            'test_loss': test_loss_history,
            'hx_history': hx_history if print_hx_every is not None else None
        }

        return epochs, history
