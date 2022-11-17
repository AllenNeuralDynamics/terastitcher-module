# Utils testing params
def get_save_dict_as_json_params():
    return [[{"number": 13, "text": "this is a test"}, "test_json.json"]]


def get_read_json_as_dict_params():
    return [
        ["test_json_2.json", {"number": 13, "text": "this is a test"}],
        ["path/does/not/exists.json", {}],
    ]


def get_helper_build_param_value_command_params():
    return [
        [
            {
                "ref1": "X",
                "ref2": "Y",
                "ref3": "D",
                "vxl1": 1.8,
                "vxl2": 1.8,
                "vxl3": 2,
                "additional_params": ["sparse_data", "libtiff_uncompress"],
            },
            "--ref1=X --ref2=Y --ref3=D --vxl1=1.8 --vxl2=1.8 --vxl3=2 ",
        ],
        [
            {
                "ref1": "X",
                "ref2": "Y",
                "ref3": "D",
                "vxl1": 1.8,
                "vxl2": 1.8,
                "vxl3": 2,
                "additional_params": ["sparse_data", "libtiff_uncompress"],
            },
            "--ref1 X --ref2 Y --ref3 D --vxl1 1.8 --vxl2 1.8 --vxl3 2 ",
            False,
        ],
        [{}, ""],
        [{}, "", False],
    ]


def get_helper_additional_params_command_params():
    return [
        [
            ["sparse_data", "libtiff_uncompress"],
            "--sparse_data --libtiff_uncompress ",
        ],
        [[], ""],
    ]


# ZarrConverter tests params
def get_compute_pyramid_params():
    return [[5, [1, 2, 2, 2]], [1, [1, 2, 2, 2]]]


def get_compute_pyramid_raises_params():
    return [[5, [1, 1, 2, 2, 2]], [5, []]]
