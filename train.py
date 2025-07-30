import shutil

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from src.definitions import AppConfig
from src.training import train


@hydra.main(
    version_base=None,
    config_path="conf/train",
    config_name=None,
)
def main(conf: AppConfig):
    hydra_conf = HydraConfig.get()
    conf.output_dir = hydra_conf.runtime.output_dir
    conf.overrides = hydra_conf.overrides.task

    # keep hydra directory config but create
    # directory manually later with additional info
    shutil.rmtree(hydra_conf.runtime.output_dir, ignore_errors=True)

    if hydra_conf.mode == RunMode.MULTIRUN:
        return conf, train

    train(conf)


if __name__ == "__main__":
    main()
