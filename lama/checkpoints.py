"""
LAMA Checkpointing Library - Do not edit here
"""

import pickle
import os
import pandas
import urllib.request
import urllib.parse
import urllib.error
import numpy as np


LAMA_CHECKPOINT_FOLDER = 'checkpoints'


def lama_compare_checkpoint(in_data, pickle_fn: str):
    # checker implementations
    def check_dataframe(in_data, golden) -> bool:
        try:
            pandas.testing.assert_frame_equal(in_data, golden, check_names=False)
        except AssertionError as e:
            raise UserWarning("Note: This only shows the first wrong column in your dataframe, there might be more. Check also the df.compare function.") from e
        return True

    def check_buildins(in_data, golden) -> bool:
        if in_data != golden:
            raise UserWarning(f"Your data does not equal the reference file. Received (type: {type(in_data)}) {in_data}, should be (type: {type(golden)}) {golden}")
        return in_data == golden

    def check_lists(in_data, golden) -> bool:
        if in_data == golden:
            return True
        else:
            # check lengths
            if len(in_data) != len(golden):
                raise UserWarning(f"Your list has an indifferent amount of elements. Yours: {len(in_data)}, should be {len(golden)}")
            else:
                # Look for wrong items
                wrong_items = [(i, a, b) for i, (a, b) in enumerate(zip(in_data, golden)) if a != b]

                error_message = f"You have {len(wrong_items)} wrong item(s) in you list:\n"
                for i, a, b in wrong_items[:10]:
                    error_message += f"\t Index {i}: Has {a}, expected {b}\n"
                if len(wrong_items) > 10:
                    error_message += "\t ... more wrong items ..."
                raise UserWarning(error_message)

    def check_ndarray(in_data, golden) -> bool:
        np.testing.assert_array_equal(in_data, golden)
        return True

    file_handlers = {
            pandas.DataFrame: check_dataframe,
            str: check_buildins,
            int: check_buildins,
            bool: check_buildins,
            list: check_lists,
            np.float64: check_buildins,
            np.float32: check_buildins,
            np.int64: check_buildins,
            np.int32: check_buildins,
            np.int8: check_buildins,
            np.ndarray: check_ndarray
            }

    # Look for local file
    d = None
    local_path = os.path.join(LAMA_CHECKPOINT_FOLDER, pickle_fn)
    if os.path.exists(local_path):
        try:
            with open(local_path, 'rb') as f:
                d = pickle.load(f)
        except IOError:
            raise AssertionError(f"Error opening offline checkpoint at {local_path}... check with your supervisor")
    else:
        try:
            fn = urllib.parse.urljoin('https://hdd1.itiv.kit.edu/tools/lama/checkpoints/', pickle_fn)
            ret = urllib.request.urlopen(fn)
            d = pickle.load(ret)
        except urllib.error.URLError:
            print("Error: Checkpoint not found locally nor online ... check with your supervisor")
            return
        except IOError:
            raise AssertionError(f"Error opening online checkpoint at {fn} ... check with your supervisor")

    if d is None:
        print("Error: Checkpoint not found locally nor online ... check with your supervisor")
        return

    if not type(in_data) in file_handlers:
        raise AssertionError(f"Datatype is not supported, maybe you gave this function the wrong input type. Received type: {type(in_data)}")

    handler_func = file_handlers[type(d)]
    ret = handler_func(in_data, d)

    if ret:
        print("Your data looks alright, you can continue with the workbook")

    return ret


def lama_create_checkpoint(data, fn, overwrite=False):
    if not os.path.exists(LAMA_CHECKPOINT_FOLDER):
        os.makedirs(LAMA_CHECKPOINT_FOLDER)

    filename = os.path.join(LAMA_CHECKPOINT_FOLDER, fn)

    if os.path.exists(filename):
        if not overwrite:
            print(f"Not creating checkpoint, file already exists at {filename}, if you want to overwrite, set the overwrite flag!")
            return
        else:
            print("Warning overwriting existing checkpoint")

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


