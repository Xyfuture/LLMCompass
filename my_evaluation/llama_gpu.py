from software_model.llama import LlamaTransformerBlockTP,ModelArgs
from software_model.utils import data_type_dict, Tensor
from hardware_model.system import system_dict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="initial computation",default=False)
    args = parser.parse_args()

    bs = 128
    s = 1024
    output_token_length = 2048

    model_args = ModelArgs()

    if args.init:
        print("Initial computation")

        model = LlamaTransformerBlockTP(
            model_args=model_args,
            device_count=4,
            data_type=data_type_dict["fp16"],
        )
        A100_system = system_dict["A100_4_fp16"]
        # from design_space_exploration.dse import read_architecture_template, template_to_system
        # arch_specs = read_architecture_template("configs/template.json")
        # A100_system = template_to_system(arch_specs)
        _ = model(Tensor([bs, s, model_args.dim], data_type_dict["fp16"]),0)

        model.compile_and_simulate(A100_system, compile_mode="heuristic-GPU")
        file_name = "llama_init_A100_sim.csv"
    else:
        print("Auto-regression")

        model = LlamaTransformerBlockTP(
            model_args = model_args,
            device_count=4,
            data_type=data_type_dict["fp16"],
        )

        A100_system = system_dict["A100_4_fp16"]
        _ = model(
            Tensor([bs, 1, model_args.dim], data_type_dict["fp16"]), s+output_token_length
        )

        model.compile_and_simulate(A100_system, compile_mode="heuristic-GPU")
        file_name = "llama_decode_A100_sim.csv"

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

