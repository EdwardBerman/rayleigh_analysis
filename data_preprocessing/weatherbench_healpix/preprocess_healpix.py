from external.custom_hermes.dataset.weatherbench_healpix import \
    WeatherbenchHealpix

era5_path = "data/weatherbench/eras5"
mesh_path = "./data/weatherbench/earth_mesh.vtp"
t850_savepath = "data/weatherbench_healpix/preprocessed/t850"
z500_savepath = 'data/weatherbench_healpix/preprocessed/z500'

WeatherbenchHealpix.preprocess_and_save(
    era5_path=era5_path,
    save_dir=t850_savepath,
    task="t850",
    nside=32,
    lmax=20,
)

WeatherbenchHealpix.preprocess_and_save(
    era5_path=era5_path,
    save_dir=z500_savepath,
    task="z500",
    nside=32,
    lmax=20
)
