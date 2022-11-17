import os
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Union

from parameterized import parameterized
from tests import params
from utils import utils

# IO types
PathLike = Union[str, Path]


class TestUtils(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        utils.save_dict_as_json(
            f"{self._tmp_dir.name}/test_json_2.json",
            {"number": 13, "text": "this is a test"},
        )

    @parameterized.expand(params.get_save_dict_as_json_params())
    def test_save_dict_as_json(self, dict_example: dict, filename: PathLike):
        path = Path(self._tmp_dir.name).joinpath(filename)
        utils.save_dict_as_json(path, dict_example)
        self.assertTrue(os.path.isfile(path))

    @parameterized.expand(params.get_read_json_as_dict_params())
    def test_read_json_as_dict(
        self, path_to_file: PathLike, expected_dict: dict
    ):
        read_dict_example = utils.read_json_as_dict(
            Path(self._tmp_dir.name).joinpath(path_to_file)
        )
        self.assertDictEqual(read_dict_example, expected_dict)

    @parameterized.expand(params.get_helper_build_param_value_command_params())
    def test_helper_build_param_value_command(
        self,
        default_config: dict,
        expected_output: str,
        equal_con: Optional[bool] = True,
    ):
        output = utils.helper_build_param_value_command(
            default_config, equal_con
        )
        self.assertEqual(expected_output, output)
        self.assertIsInstance(output, str)

    @parameterized.expand(params.get_helper_additional_params_command_params())
    def test_helper_additional_params_command(
        self, default_config: dict, expected_output: str
    ):
        output = utils.helper_additional_params_command(default_config)
        self.assertEqual(output, expected_output)
        self.assertIsInstance(output, str)

    def test_save_string_to_txt(self):
        string = "this is a message"
        path = Path(self._tmp_dir.name).joinpath("text_file.txt")
        utils.save_string_to_txt(string, path)
        self.assertTrue(os.path.isfile(path))

    def generate_timestamp(self):
        self.assertIsInstance(generate_timestamp(time_format), str)

    def tearDown(self):
        self._tmp_dir.cleanup()
