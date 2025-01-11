from transformers import TrainerCallback
import wandb

# Define a callback to record loss
# class LossRecorderCallback(TrainerCallback):
#     def __init__(self):
#         self.train_losses = []
#         self.eval_losses = []
#         self.steps = []
#         self.eval_steps = []

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             if 'loss' in logs:
#                 self.train_losses.append(logs['loss'])
#                 self.steps.append(state.global_step)
#             if 'eval_loss' in logs:
#                 self.eval_losses.append(logs['eval_loss'])
#                 self.eval_steps.append(state.global_step)

class WandbLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Log training loss and other metrics automatically handled by Trainer
        pass  # No action needed here if Trainer already logs standard metrics

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Log standard metrics
            wandb.log(logs, step=state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # Log evaluation metrics
            wandb.log(metrics, step=state.global_step)
    
    def on_train_begin(self, args, state, control, **kwargs):
        wandb.watch(kwargs.get("model"), log="all")
