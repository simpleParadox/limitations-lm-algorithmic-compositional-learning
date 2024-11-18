import os
import sys

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig

from data_modules.synthetic_tasks import (
    PointerExecutionNeighbour,
    PointerExecutionReverseMulticount,
)
from prompting_experiment_utils.prompt_builder_pen import (
    build_prompt as build_prompt_pen,
)
from prompting_experiment_utils.prompt_builder_perm import (
    build_prompt as build_prompt_perm,
)
from prompting_experiment_utils.prompt_llm import generate_text


@hydra.main(
    config_path="configs", config_name="experiment_cfg_prompting", version_base=None
)
def run_experiments(cfg: DictConfig) -> None:
    # Access the loaded config
    cfg_pen = compose(config_name="data_cfg_prompting_pen")
    cfg_pen = cfg_pen["data_mix"][0]
    cfg_perm = compose(config_name="data_cfg_prompting_perm")
    cfg_perm = cfg_perm["data_mix"][0]

    for exp_cfg in cfg["experiments"]:
        for i in range(exp_cfg["how_many_runs"]):
            make_experiment(exp_cfg, cfg_pen, cfg_perm, cfg)


def make_experiment(exp_cfg, cfg_pen, cfg_perm, full_cfg):
    task = exp_cfg["task"]
    if task == "PEN":
        sample_list = PointerExecutionNeighbour(
            min_len=cfg_pen["min_len"],
            max_len=cfg_pen["max_len"],
            min_hops=cfg_pen["min_hops"],
            max_hops=cfg_pen["max_hops"],
            sub_task=cfg_pen["sub_task"],
        ).generate(40)
        prompt, answer = build_prompt_pen(sample_list, exp_cfg)
    if task == "PERM":
        sample_list = PointerExecutionReverseMulticount(
            min_len=cfg_perm["min_len"],
            max_len=cfg_perm["max_len"],
            sub_task=cfg_perm["sub_task"],
        ).generate(40)
        prompt, answer = build_prompt_perm(sample_list, exp_cfg)
    print("\n\n")
    print(prompt)
    print(answer)
    print("\n\n")
    with_code = exp_cfg["ask_for_cot"] == "with_code"
    llm_answer = generate_text(
        prompt, credentials=full_cfg, with_code=with_code, model=full_cfg["model_name"]
    )

    print(
        "True answer: ##########################################################################"
    )
    print(answer)
    print(
        "LLM answer:  ##########################################################################"
    )
    print(llm_answer)

    here = os.path.dirname(os.path.abspath(__file__))

    folder_name = exp_cfg["save_foldername"]
    if not os.path.exists(f"{here}/{folder_name}"):
        os.makedirs(f"{here}/{folder_name}")
    filename = f'prompt_{full_cfg["model_name"]}-1.txt'
    n_obj_with_same_name = 1
    while os.path.exists(f"{here}/{folder_name}/{filename}"):
        n_obj_with_same_name += 1
        filename = f'prompt_{full_cfg["model_name"]}-{n_obj_with_same_name}.txt'
    with open(f"{here}/{folder_name}/{filename}", "w") as f:
        f.write(
            "True answer: ############################################################################################\n\n"
        )
        f.write(answer)
        f.write("\n\n")
        f.write(
            "LLM answer:  ############################################################################################\n\n"
        )
        f.write("\n")
        f.write(llm_answer)  # type: ignore
        f.write("\n\n")
        f.write(
            "Prompt:  ################################################################################################\n\n"
        )
        f.write("\n")
        f.write(prompt)
        f.write(
            "Config:  ################################################################################################\n\n"
        )
        f.write(str(exp_cfg))


if __name__ == "__main__":
    run_experiments()
