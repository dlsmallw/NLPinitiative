import unittest

from transformers import (
    PreTrainedModel,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding
)

from nlpinitiative.config import (
    MODELS_DIR
)

from nlpinitiative.modeling.predict import InferenceHandler
from nlpinitiative.modeling.train import (
    bin_train_args,
    ml_regr_train_args,
    get_bin_model,
    get_ml_model
)


class TestModelLogic(unittest.TestCase):
    def setUp(self):
        self.ih = InferenceHandler()

    def test_training_argument_creation(self):
        """Test that the training arguments are created correctly."""

        # Binary Model Arguments
        bin_args = bin_train_args()
        self.assertIsInstance(bin_args, TrainingArguments)
        self.assertEqual(bin_args.output_dir, str(MODELS_DIR / "binary_classification"))
        self.assertEqual(bin_args.eval_strategy, "steps")
        self.assertEqual(bin_args.save_strategy, "steps")
        self.assertEqual(bin_args.logging_steps, 500)
        self.assertEqual(bin_args.save_steps, 500)
        self.assertEqual(bin_args.learning_rate, 2e-5)
        self.assertEqual(bin_args.per_device_train_batch_size, 8)
        self.assertEqual(bin_args.per_device_eval_batch_size, 8)
        self.assertEqual(bin_args.num_train_epochs, 3)
        self.assertEqual(bin_args.weight_decay, 0.01)
        self.assertEqual(bin_args.load_best_model_at_end, True)
        self.assertEqual(bin_args.metric_for_best_model, "f1")
        self.assertEqual(bin_args.greater_is_better, True)
        
        ## These are preset within base function that is called
        self.assertEqual(bin_args.save_total_limit, 5)
        self.assertEqual(bin_args.save_on_each_node, True)
        self.assertEqual(bin_args.overwrite_output_dir, True)

        # Multilabel Regression Model Arguments
        ml_args = ml_regr_train_args()
        self.assertIsInstance(ml_args, TrainingArguments)
        self.assertEqual(ml_args.output_dir, str(MODELS_DIR / "multilabel_regression"))
        self.assertEqual(ml_args.eval_strategy, "steps")
        self.assertEqual(ml_args.save_strategy, "steps")
        self.assertEqual(ml_args.logging_steps, 500)
        self.assertEqual(ml_args.save_steps, 500)
        self.assertEqual(ml_args.learning_rate, 2e-5)
        self.assertEqual(ml_args.per_device_train_batch_size, 8)
        self.assertEqual(ml_args.per_device_eval_batch_size, 8)
        self.assertEqual(ml_args.num_train_epochs, 3)
        self.assertEqual(ml_args.weight_decay, 0.01)
        self.assertEqual(ml_args.load_best_model_at_end, True)
        self.assertEqual(ml_args.metric_for_best_model, "eval_mean_rmse")
        self.assertEqual(ml_args.greater_is_better, False)

        ## These are preset within base function that is called
        self.assertEqual(ml_args.save_total_limit, 5)
        self.assertEqual(ml_args.save_on_each_node, True)
        self.assertEqual(ml_args.overwrite_output_dir, True)

    def test_model_creation(self):
        """Test that the binary and multilabel regression models are created correctly."""

        # Binary Model Creation
        bin_model = get_bin_model()
        self.assertIsInstance(bin_model, PreTrainedModel)
        self.assertEqual(bin_model.config.num_labels, 2)

        # Multilabel Regression Model Creation
        ml_model = get_ml_model()
        self.assertIsInstance(ml_model, PreTrainedModel)
        self.assertEqual(ml_model.config.num_labels, 6)

    
    def test_inference_handler_initialization(self):
        """Test that the InferenceHandler is initialized correctly."""

        # Check if the tokenizers are initialized correctly
        self.assertIsInstance(self.ih.bin_tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast)
        self.assertIsInstance(self.ih.ml_regr_tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast)

        # Check if the models are initialized correctly
        self.assertIsInstance(self.ih.bin_model, PreTrainedModel)
        self.assertIsInstance(self.ih.ml_regr_model, PreTrainedModel)

    def test_inference_handler_encoding(self):
        """Test that the binary encoding function works correctly."""

        text = "This is a test sentence."
        bin_encoding, ml_encoding = self.ih.encode_input(text)

        # Check that the binary and multilabel encodings are of the correct type
        self.assertIsInstance(bin_encoding, BatchEncoding)
        self.assertIsInstance(ml_encoding, BatchEncoding)