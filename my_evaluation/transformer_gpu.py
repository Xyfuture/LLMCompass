from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from hardware_model.system import system_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="initial computation",default=False)
    args = parser.parse_args()

    bs = 128
    s = 2048
    output_token_length = 1024

    if args.init:
        print("Initial computation")

        model = TransformerBlockInitComputationTP(
            d_model=12288,
            n_heads=96,
            device_count=4,
            data_type=data_type_dict["fp16"],
        )
        A100_system = system_dict["A100_4_fp16"]
        # from design_space_exploration.dse import read_architecture_template, template_to_system
        # arch_specs = read_architecture_template("configs/template.json")
        # A100_system = template_to_system(arch_specs)
        _ = model(Tensor([bs, s, 12288], data_type_dict["fp16"]))

        model.compile_and_simulate(A100_system, compile_mode="heuristic-GPU")
        file_name = "transformer_A100_sim.csv"
    else:
        print("Auto-regression")

        model = TransformerBlockAutoRegressionTP(
            d_model=12288,
            n_heads=96,
            device_count=4,
            data_type=data_type_dict["fp16"],
        )

        A100_system = system_dict["A100_4_fp16"]
        _ = model(
            Tensor([bs, 1, 12288], data_type_dict["fp16"]), s + output_token_length
        )

        model.compile_and_simulate(A100_system, compile_mode="heuristic-GPU")
        file_name = "transformerAR_A100_sim.csv"

    simulate_dict = model.simulate_dict
    simulate_beakdown = {}
    total_latency = sum(simulate_dict.values())

    to_us = lambda x:round(x*1000000,2)

    for key,value in simulate_dict.items():
        simulate_beakdown[key] = round(value/total_latency,2)

    print("Model Info:\n"
          f"Batch Size:{bs} Sequence Length:{s+output_token_length}\n")
    for key in simulate_dict:
        print(f"{key}: {to_us(simulate_dict[key])}, {simulate_beakdown[key]*100}%")


    with open(f"./{file_name}", "w") as f:
        f.write(model.simulate_log)
