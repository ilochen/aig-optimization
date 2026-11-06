import subprocess
import tempfile
import os
import re

def run_abc(commands, abc_path=".abc/abc"):
    """
    Run ABC with a list of commands.
    :param commands: list of ABC commands (strings)
    :param abc_path: path to the abc binary
    :return: stdout output from ABC
    """
    full_script = "\n".join(commands) + "\nquit\n"

    # Run abc and pass commands via stdin
    try:
        result = subprocess.run(
            [abc_path],
            input=full_script.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode()
    except subprocess.CalledProcessError as e:
        print("ABC Error Output:")
        print(e.stderr.decode())
        raise
    
def convert_truth_to_aig(truth_path: str, output_aig: str, abc_path="./abc/abc"):
    abc_commands = [
        f"read_truth -xf {truth_path}",  # read multi-output truth table
        "collapse",                      # collapse multiple outputs into a network
        "sop",                           # factor into SOP form
        "strash",                        # convert to AIG
        "dc2",                           # optimize with don't-cares
        f"write {output_aig}",
        "print_stats"
    ]
    output = run_abc(abc_commands, abc_path)
    print(output)
    return output

def parse_abc_stats_line(line: str) -> dict:
    """
    Robustly parse ABC's print_stats output line.
    Example:
        "ex                            : i/o =   10/   10  lat =    0  and =   3620  lev = 19"
    """
    pattern = (
        r"(?P<name>\S+)\s*:\s*"
        r"i/o\s*=\s*(?P<inputs>\d+)\s*/\s*(?P<outputs>\d+)\s+"
        r"lat\s*=\s*(?P<latency>\d+)\s+"
        r"and\s*=\s*(?P<ands>\d+)\s+"
        r"lev\s*=\s*(?P<levels>\d+)"
    )
    
    match = re.search(pattern, line.strip())
    if not match:
        raise ValueError(f"Unrecognized ABC stats line format: {line!r}")

    return {
        "name": match.group("name"),
        "inputs": int(match.group("inputs")),
        "outputs": int(match.group("outputs")),
        "latency": int(match.group("latency")),
        "ands": int(match.group("ands")),
        "levels": int(match.group("levels")),
    }

def strip_ansi_escape_sequences(text: str) -> str:
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)

def detect_format(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".truth":
        return "truth"
    elif ext == ".aig":
        return "aig"
    else:
        raise ValueError(f"Unsupported benchmark file format: {ext}")


def blif_to_aig(blif_path, output_aig_path, abc_path="./abc/abc"):
    commands = [
        f"read_blif {blif_path}",
        "print_stats",
        "strash",
        f"write {output_aig_path}"
    ]
    print(f"Running ABC with commands: {commands}")
    subprocess.run(
        [abc_path],
        input="\n".join(commands).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return output_aig_path


def extract_mffc_info(aig_path, abc_path="./abc/abc"):
    commands = [
        f"read {aig_path}",
        "print_mffc"
    ]
    result = subprocess.run(
        [abc_path],
        input="\n".join(commands).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    stdout = result.stdout.decode()
    lines = stdout.strip().splitlines()

    # Updated parsing for ABC's newer output format
    mffc_info = {}
    pattern = r"Node\s*=\s*(n\d+)\s*:\s*Supp\s*=\s*(\d+)\s*Cone\s*=\s*(\d+)"

    for line in lines:
        match = re.search(pattern, line)
        if match:
            node_name = match.group(1)
            support = int(match.group(2))
            cone = int(match.group(3))
            mffc_info[node_name] = {
                "mffc_size": cone,
                "support_size": support
            }

    return mffc_info

import networkx as nx

def parse_blif_with_truth_tables(blif_path):
    import networkx as nx
    G = nx.DiGraph()
    logic_lines = []
    current_block = []

    with open(blif_path, "r") as f:
        for line in f:
            if line.startswith(".names"):
                if current_block:
                    logic_lines.append(current_block)
                current_block = [line]
            elif current_block and (line.startswith(".") or line.strip() == ""):
                logic_lines.append(current_block)
                current_block = []
            elif current_block:
                current_block.append(line)
        if current_block:
            logic_lines.append(current_block)

    structured_logic = []
    for block in logic_lines:
        header = block[0].strip().split()
        *fanins, out = header[1:]
        structured_logic.append((fanins, out, block))

        for fi in fanins:
            G.add_edge(fi, out)

    return G, structured_logic



def extract_cone_graph(G, target_node):
    cone_nodes = nx.ancestors(G, target_node) | {target_node}
    return G.subgraph(cone_nodes).copy()


def write_exact_cone_to_blif(cone_graph, all_logic_lines, root_node, output_path):
    # Extract only .names blocks that are part of the cone
    cone_nodes = set(cone_graph.nodes)
    inputs = [n for n in cone_nodes if cone_graph.in_degree(n) == 0]

    with open(output_path, "w") as f:
        f.write(".model exact_cone\n")
        f.write(".inputs " + " ".join(inputs) + "\n")
        f.write(".outputs " + root_node + "\n")

        for fanins, out, lines in all_logic_lines:
            if out in cone_nodes and all(fi in cone_nodes for fi in fanins):
                for l in lines:
                    f.write(l)

        f.write(".end\n")
    return output_path



def synthesize_blif_with_abc(blif_in, blif_out, abc_path="./abc/abc"):
    commands = [
        f"read_blif {blif_in}",
        "strash",
        "balance",
        "rewrite",
        "resub",
        "dc2",
        f"write_blif {blif_out}"
    ]
    subprocess.run(
        [abc_path],
        input="\n".join(commands).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

def reintegrate_cone_fixed(
    original_blif_path,
    optimized_blif_path,
    cone_nodes,
    output_blif_path
):
    with open(original_blif_path, "r") as f:
        original_lines = f.readlines()

    with open(optimized_blif_path, "r") as f:
        opt_lines = f.readlines()

    # Extract optimized .names blocks
    optimized_blocks = []
    current_block = []
    for line in opt_lines:
        if line.startswith(".names"):
            if current_block:
                optimized_blocks.append(current_block)
            current_block = [line]
        elif current_block and (line.startswith(".") or line.strip() == ""):
            optimized_blocks.append(current_block)
            current_block = []
        elif current_block:
            current_block.append(line)
    if current_block:
        optimized_blocks.append(current_block)

    # Parse and remove only blocks whose output is in cone_nodes
    new_blif_lines = []
    current_block = []
    skip_block = False

    for line in original_lines:
        if line.startswith(".names"):
            if current_block and not skip_block:
                new_blif_lines.extend(current_block)
            current_block = [line]
            tokens = line.strip().split()
            output = tokens[-1]
            skip_block = output in cone_nodes
        elif current_block:
            if line.startswith(".") or line.strip() == "":
                if not skip_block:
                    current_block.append(line)
                    new_blif_lines.extend(current_block)
                current_block = []
                skip_block = False
            else:
                current_block.append(line)
        else:
            new_blif_lines.append(line)

    if current_block and not skip_block:
        new_blif_lines.extend(current_block)

    # Insert optimized blocks before .end
    end_index = next(i for i, l in enumerate(new_blif_lines) if l.strip() == ".end")
    for block in optimized_blocks:
        new_blif_lines[end_index:end_index] = block
        end_index += len(block)


    with open(output_blif_path, "w") as f:
        f.writelines(new_blif_lines)

    return output_blif_path



def check_aig_equivalence(aig1, aig2, abc_path="./abc/abc", match_by_order=True):
    flag = "-n" if match_by_order else ""
    commands = [
        f"read {aig1}",
        f"cec {flag} {aig2}".strip()
    ]
    result = subprocess.run(
        [abc_path],
        input="\n".join(commands).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return result.stdout.decode()




# Example usage
if __name__ == "__main__":
    # abc_raw_output = convert_truth_to_aig("./2025_IWLS_Contest_Benchmarks_020425/ex100.truth", "output.aig")
    # stats_line = [line for line in abc_raw_output.strip().splitlines() if "i/o" in line][-1]
    # #stats_line = [line for line in abc_output.strip().splitlines() if "i/o" in line][-1]
    # stats_line = strip_ansi_escape_sequences(stats_line)
    # stats = parse_abc_stats_line(stats_line)
    # info = extract_mffc_info("runtime_results/current.aig")
    # for node, props in list(info.items())[:10]:
    #     print(f"Node {node}: MFFC={props['mffc_size']}, Support={props['support_size']}")
    # G, all_logic = parse_blif_with_truth_tables("runtime_results/current.blif")
    # cone = extract_cone_graph(G, "new_n1085")
    # write_exact_cone_to_blif(cone, all_logic, "new_n1085", "runtime_results/cone.blif")

    # # synthesize_blif_with_abc("runtime_results/cone.blif", "runtime_results/cone_opt.blif")
    # reintegrate_cone_fixed(
    #     original_blif_path="runtime_results/current.blif",
    #     optimized_blif_path="runtime_results/cone_opt.blif",
    #     cone_nodes=set(cone.nodes),
    #     output_blif_path="runtime_results/final_cleaned.blif"
    # )


    blif_to_aig("runtime_results/final_cleaned.blif", "runtime_results/final_cleaned.aig")
    print(check_aig_equivalence("runtime_results/current.aig", "runtime_results/final_cleaned.aig"))



    # print(stats)
