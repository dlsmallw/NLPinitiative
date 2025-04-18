import os
import json
import unittest
import pandas as pd
from datasets import DatasetDict
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)

from nlpinitiative.config import TEST_DATA_DIR
from nlpinitiative.data_preparation.data_management import DataManager, DatasetContainer

class TestDataManager(unittest.TestCase):
    def setUp(self):
        """Set up the test case by creating a DataManager instance and generating test files."""
        self.dm = DataManager()

        test_dataset1 = pd.DataFrame(
            data=[
                ['sample1', 0.45, 0.78, 0.12, 0.34, 0.56, 0.78],
                ['sample2', 0.67, 0.34, 0.89, 0.12, 0.45, 0.67],
                ['sample3', 0.23, 0.56, 0.91, 0.34, 0.78, 0.12],
                ['sample4', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            columns=['text','value1','value2','value3','value4','value5','value6']
        )
        test_dataset1.to_csv(TEST_DATA_DIR / "test_data1.csv", index=False)
        test_dataset1.to_csv(TEST_DATA_DIR / "test_data_semicolon_delimiter.csv", sep=';', index=False)
        with open(TEST_DATA_DIR / 'test_data1.json', 'w') as f:
            json.dump([
                {"text": "sample1","value1": 0.45,"value2": 0.78,"value3": 0.12,"value4": 0.34,"value5": 0.56,"value6": 0.78},
                {"text": "sample2","value1": 0.67,"value2": 0.34,"value3": 0.89,"value4": 0.12,"value5": 0.45,"value6": 0.67},
                {"text": "sample3","value1": 0.23,"value2": 0.56,"value3": 0.91,"value4": 0.34,"value5": 0.78,"value6": 0.12},
                {"text": "sample4","value1": 0.0,"value2": 0.0,"value3": 0.0,"value4": 0.0,"value5": 0.0,"value6": 0.0}
            ], f)
            f.close()
        test_dataset1.to_excel(TEST_DATA_DIR / "test_data1.xlsx", index=False)

        pd.DataFrame(
            data=[
                ['sample1',1],
                ['sample2',1],
                ['sample3',1],
                ['sample4',0]
            ],
            columns=['text','random_value']
        ).to_csv(TEST_DATA_DIR / "test_data2.csv", index=False)
        
        pd.DataFrame(
            data=[
                ['sample1','religion'],
                ['sample2','disability'],
                ['sample3','sexuality'],
                ['sample4','racism'],
                ['sample5','gender']
            ],
            columns=['text','classification']
        ).to_csv(TEST_DATA_DIR / "test_data3.csv", index=False)

        with open(TEST_DATA_DIR / 'test_norm_schema1.json', 'w') as f:
            json.dump({
                "data_col": "text",
                "mapping_type": "many2many",
                "single_column_label": None,
                "column_mapping": {
                    "DISCRIMINATORY": ["random_value"],
                    "GENDER": ["value1"],
                    "RACE": ["value2"],
                    "SEXUALITY": ["value3"],
                    "DISABILITY": ["value4"],
                    "RELIGION": ["value5"],
                    "UNSPECIFIED": ["value6"]
                }
            }, f)
            f.close()

        with open(TEST_DATA_DIR / 'test_norm_schema2.json', 'w') as f:
            json.dump({
                "data_col": "text",
                "mapping_type": "one2many",
                "single_column_label": "classification",
                "column_mapping": {
                    "GENDER": ["gender"],
                    "RACE":  ["racism"],
                    "SEXUALITY": ["sexuality"],
                    "DISABILITY": ["disability"],
                    "RELIGION":  ["religion"],
                    "UNSPECIFIED": []
                }
            }, f)
            f.close()

        self.dm.normalize_dataset(ds_files=["test_data1.csv", "test_data2.csv"], conv_schema_fn="test_norm_schema1.json", output_fn="norm_ds_for_prep", testing=True)

    def tearDown(self):
        """Clean up the test case by removing generated files."""
        for f in os.listdir(TEST_DATA_DIR):
            if not os.path.isdir(f):
                self.dm.remove_file(filename=f, path=TEST_DATA_DIR)

        for f in os.listdir(TEST_DATA_DIR / "output"):
            if f != ".gitkeep":
                self.dm.remove_file(filename=f, path=TEST_DATA_DIR / "output")

    def test_url_validation(self):
        """Test the URL validation function."""

        # Test with a valid URL and an invalid URL
        valid_url = "https://example.com/data"
        invalid_url = "invalid_url"
        self.assertTrue(self.dm._is_valid_url(valid_url))
        self.assertFalse(self.dm._is_valid_url(invalid_url))

        # Test with a valid GitHub URLs
        valid_gh_url_one = "https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv"
        valid_gh_url_two = "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv"
        self.assertTrue(self.dm._is_valid_url(valid_gh_url_one))
        self.assertTrue(self.dm._is_valid_url(valid_gh_url_two))
    
    def test_import_filename_generation(self):
        """Test the filename generation function."""

        gh_url = "https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv"
        non_gh_url = "https://example.com/data"
        expected_gh_filename = "intelligence-csd-auth-gr_Ethos-Hate-Speech-Dataset_Ethos_Dataset_Binary"
        expected_non_gh_filename = "data"
        self.assertEqual(self.dm._generate_import_filename(gh_url), expected_gh_filename)
        self.assertEqual(self.dm._generate_import_filename(non_gh_url), expected_non_gh_filename)

    def test_file_to_df(self):
        """Test the file to DataFrame conversion function."""

        # Test with a valid CSV file
        csv1_file_path = TEST_DATA_DIR / "test_data1.csv"
        df1 = self.dm.file_to_df(source=csv1_file_path, ext=".csv")
        self.assertIsInstance(df1, pd.DataFrame)
        self.assertFalse(df1.empty)

        # Test with a valid CSV file with semi-colon delimiter
        csv2_file_path = TEST_DATA_DIR / "test_data_semicolon_delimiter.csv"
        df2 = self.dm.file_to_df(source=csv2_file_path, ext=".csv")
        self.assertIsInstance(df2, pd.DataFrame)
        self.assertFalse(df2.empty)

        # Test with a valid JSON file
        json_file_path = TEST_DATA_DIR / "test_data1.json"
        df3 = self.dm.file_to_df(source=json_file_path, ext=".json")
        self.assertIsInstance(df3, pd.DataFrame)
        self.assertFalse(df3.empty)

        # Test with a valid Excel file
        excel_file_path = TEST_DATA_DIR / "test_data1.xlsx"
        df4 = self.dm.file_to_df(source=excel_file_path, ext=".xlsx")
        self.assertIsInstance(df4, pd.DataFrame)
        self.assertFalse(df4.empty)

        # Test with a non-existent file
        non_existent_file_path = TEST_DATA_DIR / "non_existent_file.csv"
        with self.assertRaises(FileNotFoundError):
            self.dm.file_to_df(non_existent_file_path, ext=".csv")

    def test_import_data(self):
        """Test the import data function."""

        dest_path = TEST_DATA_DIR / 'output';

        # Local Import Testing
        ## Test with a invalid csv file
        with self.assertRaises(expected_exception=FileNotFoundError):
            local_result = self.dm.import_data(
                import_type='local', 
                source=(TEST_DATA_DIR / "invalid_test_data.csv"),
                dataset_name='local_import_test',  
                output_fn='local_import_test',
                destination=dest_path,
                local_ds_ref_url="https://example.com/data",
                testing=True
            )

        ## Test without a reference url
        with self.assertRaises(expected_exception=Exception):
            local_result = self.dm.import_data(
                import_type='local', 
                source=(TEST_DATA_DIR / "test_data1.csv"),
                dataset_name='local_import_test',  
                output_fn='local_import_test',
                destination=dest_path,
                testing=True
            )
        
        ## Test with valid reference url
        local_result = self.dm.import_data(
            import_type='local', 
            source=(TEST_DATA_DIR / "test_data1.csv"),
            dataset_name='local_import_test',
            destination=dest_path,
            local_ds_ref_url="https://example.com/data",
            testing=True
        )
        self.assertIsInstance(local_result, pd.DataFrame)
        self.assertTrue(os.path.exists(dest_path / "test_data1.csv"))
        self.dm.remove_file(filename="test_data1.csv", path=dest_path)

        # Remote Import Testing
        ## Test with invalid file type
        with self.assertRaises(expected_exception=Exception):
            remote_result = self.dm.import_data(
                import_type='external', 
                source='https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.docx',
                dataset_name='remote_import_test',  
                output_fn='remote_import_test',
                destination=dest_path,
                testing=True
            )

        ## Test with invalid url
        with self.assertRaises(expected_exception=Exception):
            remote_result = self.dm.import_data(
                import_type='external', 
                source="invalid_url",
                dataset_name='remote_import_test',  
                output_fn='remote_import_test',
                destination=dest_path,
                testing=True
            )

        ## Test with valid url
        remote_result = self.dm.import_data(
            import_type='external', 
            source='https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv',
            dataset_name='remote_import_test',  
            output_fn='remote_import_test',
            destination=dest_path,
            testing=True
        )
        self.assertIsInstance(remote_result, pd.DataFrame)
        self.assertTrue(os.path.exists(dest_path / "remote_import_test.csv"))
        self.dm.remove_file(filename="remote_import_test.csv", path=dest_path)

    def test_dataset_normalization(self):
        """Test the dataset normalization function."""

        # Test with invalid files
        with self.assertRaises(expected_exception=FileNotFoundError):
            normalized_df = self.dm.normalize_dataset(
                ds_files=["invalid_test_data.csv", "test_data2.csv"], 
                conv_schema_fn='test_norm_schema1.json', 
                output_fn="normalized_test_data", 
                testing=True
            )

        # Test with invalid conversion schema filename
        with self.assertRaises(expected_exception=FileNotFoundError):
            normalized_df = self.dm.normalize_dataset(ds_files=["test_data1.csv"], conv_schema_fn="invalid_norm_schema.json", output_fn="normalized_test_data", testing=True)

        # Test with invalid output filename
        with self.assertRaises(expected_exception=Exception):
            normalized_df = self.dm.normalize_dataset(ds_files=["test_data1.csv"], conv_schema_fn="test_norm_schema1.json", output_fn="", testing=True)

        # Test with valid file containing one to many relationships
        eval_df2 = pd.DataFrame(
            data=[
                ['sample1', 1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ['sample2', 1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ['sample3', 1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ['sample4', 1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ['sample5', 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            columns=['TEXT','DISCRIMINATORY','GENDER','RACE','SEXUALITY','DISABILITY','RELIGION','UNSPECIFIED']
        )
        normalized_df = self.dm.normalize_dataset(ds_files=["test_data3.csv"], conv_schema_fn="test_norm_schema2.json", output_fn="normalized_test_data", testing=True)
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertFalse(normalized_df.empty)
        for index, row in normalized_df.iterrows():
            self.assertEqual(row['TEXT'], eval_df2.iloc[index]['TEXT'])
            self.assertEqual(row['DISCRIMINATORY'], eval_df2.iloc[index]['DISCRIMINATORY'])
            self.assertEqual(row['GENDER'], eval_df2.iloc[index]['GENDER'])
            self.assertEqual(row['RACE'], eval_df2.iloc[index]['RACE'])
            self.assertEqual(row['SEXUALITY'], eval_df2.iloc[index]['SEXUALITY'])
            self.assertEqual(row['DISABILITY'], eval_df2.iloc[index]['DISABILITY'])
            self.assertEqual(row['RELIGION'], eval_df2.iloc[index]['RELIGION'])
            self.assertEqual(row['UNSPECIFIED'], eval_df2.iloc[index]['UNSPECIFIED'])
        self.dm.remove_file(filename="normalized_test_data.csv", path=(TEST_DATA_DIR / "output"))

        # Test with valid files containing many to many relationships
        eval_df1 = pd.DataFrame(
            data=[
                ['sample1',1,0.45,0.78,0.12,0.34,0.56,0.78],
                ['sample2',1,0.67,0.34,0.89,0.12,0.45,0.67],
                ['sample3',1,0.23,0.56,0.91,0.34,0.78,0.12],
                ['sample4',0,0.0,0.0,0.0,0.0,0.0,0.0]
            ],
            columns=['TEXT','DISCRIMINATORY','GENDER','RACE','SEXUALITY','DISABILITY','RELIGION','UNSPECIFIED']
        )
        normalized_df = self.dm.normalize_dataset(ds_files=["test_data1.csv", "test_data2.csv"], conv_schema_fn="test_norm_schema1.json", output_fn="normalized_test_data", testing=True)
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertFalse(normalized_df.empty)
        for index, row in normalized_df.iterrows():
            self.assertEqual(row['TEXT'], eval_df1.iloc[index]['TEXT'])
            self.assertEqual(row['DISCRIMINATORY'], eval_df1.iloc[index]['DISCRIMINATORY'])
            self.assertEqual(row['GENDER'], eval_df1.iloc[index]['GENDER'])
            self.assertEqual(row['RACE'], eval_df1.iloc[index]['RACE'])
            self.assertEqual(row['SEXUALITY'], eval_df1.iloc[index]['SEXUALITY'])
            self.assertEqual(row['DISABILITY'], eval_df1.iloc[index]['DISABILITY'])
            self.assertEqual(row['RELIGION'], eval_df1.iloc[index]['RELIGION'])
            self.assertEqual(row['UNSPECIFIED'], eval_df1.iloc[index]['UNSPECIFIED'])

        self.dm.remove_file(filename="normalized_test_data.csv", path=(TEST_DATA_DIR / "output"))

    def test_data_prep_and_processing(self):
        """Test the data preparation and processing function."""

        ds_obj_one, ds_obj_two = self.dm.prepare_and_preprocess_dataset(
            filename="norm_ds_for_prep.csv",
            srcdir=(TEST_DATA_DIR / "output")
        )

        self.assertIsNotNone(ds_obj_one)
        self.assertIsNotNone(ds_obj_two)
        self.assertIsInstance(ds_obj_one, DatasetContainer)
        self.assertIsInstance(ds_obj_two, DatasetContainer)
        self.assertIsInstance(ds_obj_one.raw_dataset, DatasetDict)
        self.assertIsInstance(ds_obj_two.raw_dataset, DatasetDict)
        self.assertIsInstance(ds_obj_one.encoded_dataset, DatasetDict)
        self.assertIsInstance(ds_obj_two.encoded_dataset, DatasetDict)
        self.assertIsInstance(ds_obj_one.tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast)
        self.assertIsInstance(ds_obj_two.tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast)
        self.assertIsInstance(ds_obj_one.labels, list)
        self.assertIsInstance(ds_obj_two.labels, list)
        self.assertIsInstance(ds_obj_one.lbl2idx, dict)
        self.assertIsInstance(ds_obj_two.lbl2idx, dict)
        self.assertIsInstance(ds_obj_one.idx2lbl, dict)
        self.assertIsInstance(ds_obj_two.idx2lbl, dict)

    def test_dataset_statisitics_creation(self):
        """Test the dataset statistics creation function."""
        
        stats = self.dm.get_dataset_statistics(
            ds_path=(TEST_DATA_DIR / "output" / "norm_ds_for_prep.csv")
        )

        # General statistics
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)
        self.assertIn("total_num_entries", stats)
        self.assertEqual(stats["total_num_entries"], 4)
        self.assertIn("num_positive_discriminatory", stats)
        self.assertEqual(stats["num_positive_discriminatory"], 3)
        self.assertIn("num_negative_discriminatory", stats)
        self.assertEqual(stats["num_negative_discriminatory"], 1)

        # Category statistics
        self.assertIn("category_stats", stats)
        self.assertIsInstance(stats["category_stats"], dict)
        self.assertEqual(len(stats["category_stats"].keys()), 6)

        for cat in stats["category_stats"].keys():
            self.assertIn("total_combined_value", stats["category_stats"][cat])
            self.assertIn("num_positive_rows", stats["category_stats"][cat])
            self.assertIn("num_gt_threshold", stats["category_stats"][cat])
            self.assertIn("num_rows_as_dominant_category", stats["category_stats"][cat])
            self.assertTrue(stats["category_stats"][cat]["total_combined_value"] > 0.0)
            self.assertEqual(stats["category_stats"][cat]["num_positive_rows"], 3)
    
            
