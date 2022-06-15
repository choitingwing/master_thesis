"""
    Call using Dataset(dataset_name, em, noise) with
    dataset_name:
        ALVAREZ (only had + noise) / ARZ
    em:
        True / False (default)
    noise:
        True (default) / False
"""

BASE_PATH = "/mnt/md0/data/SouthPole/single_surface_4LPDA_PA_15m_RNOG_fullsim.json"

class Dataset:
    def __init__(self, dataset_name, em=False, noise=True):
        if dataset_name.upper() == "ALVAREZ":
            if em:
                raise ValueError("'ALVAREZ' dataset does not have em=True")
            if not noise:
                raise ValueError("'ALVAREZ' dataset does not have noise=False")

            self.test_file_ids = [80, 81, 82]
            self.datapath = f"{BASE_PATH}/Alvarez2009_had_noise.yaml/G03generate_events_full_surface_sim/v2/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/"
            self.data_filename = "data_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
            self.label_filename = "labels_1-3_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
            self.n_files = 82
            self.n_files_val = 10
        elif dataset_name.upper() == "ARZ":
            if noise:
                noise_string = "em_had_separately"
            else:
                noise_string = "noiseless"

            if em:
                self.test_file_ids = [47, 48, 49]
                self.datapath = f"{BASE_PATH}/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/{noise_string}/"
                self.data_filename = "data_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
                self.label_filename = "labels_emhad_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
                self.n_files = 50
                self.n_files_val = 6
            else:
                self.test_file_ids = [38, 39, 40]
                self.datapath = f"{BASE_PATH}/ARZ2020_emhad_noise.yaml/G03generate_events_full_surface_sim/LPDA_2of4_100Hz/4LPDA_1dipole_fullband/{noise_string}/"
                self.data_filename = "data_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
                self.label_filename = "labels_had_emhad_1-3_had_1_LPDA_2of4_100Hz_4LPDA_1dipole_fullband_"
                self.n_files = 41
                self.n_files_val = 5
        else:
            raise ValueError(f"dataset_name ({dataset_name.upper()}) must be either 'ALVAREZ' or 'ARZ'")
