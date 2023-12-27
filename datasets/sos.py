from .base import DatasetBase
import os 
from typing import Optional,Tuple
import pandas as pd
import numpy as np
from utils import logger,cal_snr
from ._factory import register_dataset



class SOS(DatasetBase):
    """Waveform from sos"""
    
    _name = "sos"
    _part_range = None
    _channels = ["z"]
    _sampling_rate = 500
    
    def __init__(
        self,
        seed:int,
        mode:str,
        data_dir:str,
        shuffle:bool=True,
        data_split:bool=False,
        train_size:float=0.8,
        val_size:float=0.1,
        **kwargs
        ):
        
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )
        
    
    def _load_meta_data(self)->pd.DataFrame:
        if self._data_split:
            logger.warning(
                f"dataset 'sos' has been split into 'train','val' and 'test', argument 'data_split' will be ignored."
            )

        csv_path = os.path.join(self._data_dir, self._mode, "_all_label.csv")
        meta_df = pd.read_csv(
            csv_path, dtype={"fname": str, "itp": int, "its": int}
        )
        
        return meta_df
    
    
    def _load_event_data(self,idx:int) -> Tuple[dict,dict]:
        """Load event data

        Args:
            idx (int): Index of target row.

        Returns:
            dict: Data of event.
        """  
        target_event = self._meta_data.iloc[idx]
        
        fname = target_event["fname"]
        ppk = target_event["itp"]
        spk = target_event["its"]

        fpath = os.path.join(self.data_dir, self.mode, fname)

        npz = np.load(fpath)

        data = npz["data"].astype(np.float32)
        
        data = np.stack(data, axis=1)

        event = {
            "data": data,
            "ppks": [ppk] if ppk > 0 else [],
            "spks": [spk] if spk > 0 else [],
            "snr": cal_snr(data=data,pat=ppk) if ppk > 0 else 0.
        }
        
        return event,target_event.to_dict()
    
@register_dataset
def sos(**kwargs):
    dataset = SOS(**kwargs)
    return dataset