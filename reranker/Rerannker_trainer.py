import os
import logging
from typing import Optional
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class RerankTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save = model_to_save.model
        model_to_save.save_pretrained(
            output_dir, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)