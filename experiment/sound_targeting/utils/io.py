import h5py
import json
import warnings
import pathlib
import numpy as np
import pandas as pd


def expandhome(path):
    """ Return the full file path including home directory (expand '~')
    """
    return str(pathlib.Path(path).expanduser())


def convert_numpy_to_native(data):
    """ Converts numpy types in a nested dictionary or list tree to native Python types
    """
    if isinstance(data, (dict, list)):
        for k, v in (data.items() if isinstance(data, dict) else enumerate(data)):
            if isinstance(v, (dict, list)):
                data[k] = convert_numpy_to_native(v)
            elif isinstance(v, np.generic):
                data[k] = v.item()
    return data


def save_to_h5(filename, data, serialize=True, compression=None, json_compression='gzip', verbosity=1,
               file_mode='w', convert_numpy_to_native=False):
    '''
    Save a nested dictionary data structure to an HDF5 file.

    Args:
        filename (string): file name of the HDF5 file
        data (dict): Nested dictionary whose contents may be dict, ndarray, str, bytes, DataFrame and JSON-serializable objects
        serialize (boolean): enable JSON serialization
        compression (string): h5py compression type (e.g. 'gzip', 'lzf' or None)
        json_compression (string): h5py compression type for serialized JSON (default: 'gzip')
        file_mode (string): h5py.File access mode. 'w' (default) for create/detete and 'a' for create/append

    based on https://github.com/danionella/lib2p/blob/master/lib2putils.py
    '''

    def recursively_save_contents_to_group(h5file, path, data_item):
        assert isinstance(data_item, (dict))
        for key, item in data_item.items():
            if verbosity > 1:
                print('saving entry: {} -- {}'.format(path + key, type(item)))
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, float)):
                comp = None if np.isscalar(item) else compression
                try:
                    h5file[path].create_dataset(key, data=item, compression=comp)
                except TypeError:
                    warnings.warn(
                        f'\n\nkey: {key} -- Saving data with compression failed. Saving without compression.\n')
                    h5file[path].create_dataset(key, data=item, compression=None)
            elif isinstance(item, pd.DataFrame):
                json_bytes = np.frombuffer(item.to_json().encode('utf-8'), dtype='byte')
                h5file[path].create_dataset(key, data=json_bytes, compression=json_compression)
                h5file[path + key].attrs[
                    'pandas_json_type'] = f'This {type(item)} was JSON serialized and UTF-8 encoded.'
            elif isinstance(item, dict):
                h5file[path].create_group(key)
                recursively_save_contents_to_group(h5file, path + key + '/', item)
            elif serialize:
                if verbosity > 0:
                    print(f'serializing {type(item)} at {path + key}', flush=True)
                json_bytes = json.dumps(item).encode('utf-8')
                h5file[path].create_dataset(key, data=np.frombuffer(json_bytes, dtype='byte'),
                                            compression=json_compression)
                h5file[path + key].attrs['json_type'] = f'This {type(item)} was JSON serialized and UTF-8 encoded.'
            else:
                raise ValueError(f'Cannot save {type(item)} to {path + key}. Consider enabling serialisation.')

    if convert_numpy_to_native:
        data = convert_numpy_to_native(data)

    filename = expandhome(filename)
    with h5py.File(filename, file_mode) as h5file:
        recursively_save_contents_to_group(h5file, '/', data)


def load_from_h5(filename):
    '''
    Load an HDF5 file to a dictionary

    Args:
        filename (string): file name of the HDF5 file

    Returns:
        dict: file contents
    '''

    def recursively_load_contents_from_group(h5file, path):
        ans = dict()
        for key, item in h5file[path].items():
            if 'pandas_type' in item.attrs.keys():
                ans[key] = pd.read_hdf(filename, path + key)
            elif 'pandas_json_type' in item.attrs.keys():
                json_str = item[()].tobytes().decode('utf-8')
                ans[key] = pd.read_json(json_str)
            elif 'json_type' in item.attrs.keys():
                ans[key] = json.loads(item[()].tobytes())
            elif isinstance(item, h5py._hl.dataset.Dataset):
                if h5py.check_string_dtype(item.dtype) is not None:
                    item = item.asstr()
                ans[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_contents_from_group(h5file, path + key + '/')
            else:
                raise ValueError(f"I don't know what to do about {path + key}.")
        return ans

    filename = expandhome(filename)
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_contents_from_group(h5file, '/')
