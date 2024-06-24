import argparse
import json
import os.path
import random
import sys

import numpy as np
import torch
import pandas as pd
from data_utils import (
    element_dict_rev,
    alphabet,
    restype_int_to_str,
    featurize,
    parse_PDB,
)
from model_utils import ProteinMPNN
from loguru import logger

def ligmpnn_score(args, cache_dir) -> None:
    """
    Inference function
    """

    seed = int(args.RNG_seed)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (int(args.GPUs) >=1) else "cpu")
    folder_for_outputs = os.path.join(args.outdir, f'{args.name}_ProteinMPNN_output')
    base_folder = folder_for_outputs
    # if base_folder[-1] != "/":
    #     base_folder = base_folder + "/"
    # if not os.path.exists(base_folder):
    #     os.makedirs(base_folder, exist_ok=True)
    # if not os.path.exists(base_folder + "seqs"):
    #     os.makedirs(base_folder + "seqs", exist_ok=True)
    # if not os.path.exists(base_folder + "backbones"):
    #     os.makedirs(base_folder + "backbones", exist_ok=True)
    # if not os.path.exists(base_folder + "packed"):
    #     os.makedirs(base_folder + "packed", exist_ok=True)
    # # if args.save_stats:
    # if not os.path.exists(base_folder + "stats"):
    #     os.makedirs(base_folder + "stats", exist_ok=True)
    # if args.model_type == "protein_mpnn":
    #     checkpoint_path = args.checkpoint_protein_mpnn
    # elif args.model_type == "ligand_mpnn":
    #     checkpoint_path = args.checkpoint_ligand_mpnn
    # elif args.model_type == "per_residue_label_membrane_mpnn":
    #     checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    # elif args.model_type == "global_label_membrane_mpnn":
    #     checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    # elif args.model_type == "soluble_mpnn":
    #     checkpoint_path = args.checkpoint_soluble_mpnn
    # else:
    #     print("Choose one of the available models")
    #     sys.exit()


    mpnn_model_dict = {'Local_Membrane' : 'per_residue_label_membrane_mpnn_v_48_020',
                        'Global_Membrane' : 'global_label_membrane_mpnn_v_48_020',
                        'Side-Chain_Packing' : 'ligandmpnn_sc_v_32_002_16',
                        'Soluble' : f'solublempnn_v_48_{args.lig_mpnn_noise}',
                        'ProteinMPNN' : f'proteinmpnn_v_48_{args.lig_mpnn_noise}',
                        'LigandMPNN' : f'ligandmpnn_v_32_{args.lig_mpnn_noise}_25.pt'
                        }
    checkpoint_path = os.path.join(cache_dir, f'LigandMPNN_weights/{mpnn_model_dict[args.mpnn_model]}.pt')
    atom_context_num = 1
    ligand_mpnn_use_side_chain_context = 0
    if "soluble" in f'{mpnn_model_dict[args.mpnn_model]}':
        args.model_type = "soluble_mpnn"
    elif "global" in f'{mpnn_model_dict[args.mpnn_model]}':
        args.model_type = "global_label_membrane_mpnn"   
    elif "residue" in f'{mpnn_model_dict[args.mpnn_model]}':
        args.model_type = "per_residue_label_membrane_mpnn"     
    elif "protein" in f'{mpnn_model_dict[args.mpnn_model]}':
        args.model_type = 'protein_mpnn'
    else:
        args.model_type = "ligand_mpnn"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if (args.transmembrane_buried or args.transmembrane_interface) and args.model_type != "per_residue_label_membrane_mpnn":
        logger.error("You need to specify --mpnn_model Local_Membrane to specify buried/interface residues!")
        raise Exception("You need to specify --mpnn_model Local_Membrane to specify buried/interface residues!")

    if args.query.endswith('.txt') and (args.transmembrane_buried or args.transmembrane_interface):
        if type(args.transmembrane_buried) == list or type(args.transmembrane_interface) == list:
            logger.warning('You have specified multiple proteins for scoring but only one set of transmembrane residues. Did you mean to do this?')
            
    k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.query.endswith('.txt'):
        with open(args.query, "r") as fh:
            pdb_paths = fh.read().splitlines()
    else:
        pdb_paths = [args.query]
    
        

    names = []
    autoreg_scores = []
    if args.batch_transmembrane_csv:
        transmem_df = pd.read_csv(args.batch_transmembrane_csv)
        for pdb, (_, row) in zip(pdb_paths, transmem_df.iterrows()):

            transmembrane_buried = row['transmembrane_buried'].split() if pd.notna(row['transmembrane_buried']) else []
            transmembrane_interface = row['transmembrane_interface'].split() if pd.notna(row['transmembrane_interface']) else []

            # Parse the PDB file
            protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                pdb,
                device=device,
                chains="",
                parse_all_atoms=0,
                parse_atoms_with_zero_occupancy=0
            )

            # Make chain_letter + residue_idx + insertion_code mapping to integers
            R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
            chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
            encoded_residues = []
            for i, R_idx_item in enumerate(R_idx_list):
                tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
                encoded_residues.append(tmp)
            encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
            encoded_residue_dict_rev = dict(
                zip(list(range(len(encoded_residues))), encoded_residues)
            )

            # Determine buried positions
            buried_positions = torch.tensor(
                [int(item in transmembrane_buried) for item in encoded_residues],
                device=device,
            ) if transmembrane_buried else torch.zeros(len(encoded_residues), device=device)

            # Determine interface positions
            interface_positions = torch.tensor(
                [int(item in transmembrane_interface) for item in encoded_residues],
                device=device,
            ) if transmembrane_interface else torch.zeros(len(encoded_residues), device=device)

            # Set membrane per residue labels
            protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
                1 - interface_positions
            ) + 1 * interface_positions * (1 - buried_positions)

            # Set global label if model type is global_label_membrane_mpnn
            if args.model_type == "global_label_membrane_mpnn":
                protein_dict["membrane_per_residue_labels"] = torch.full(
                    (len(encoded_residues),), args.global_transmembrane_label, device=device
                )

            chains_to_design_list = protein_dict["chain_letters"]
            chain_mask = torch.tensor(
                np.array(
                    [
                        item in chains_to_design_list
                        for item in protein_dict["chain_letters"]
                    ],
                    dtype=np.int32,
                ),
                device=device,
            )
            protein_dict["chain_mask"] = chain_mask
            remapped_symmetry_residues = [[]]
            name = pdb[pdb.rfind("/") + 1 :]
            if name[-4:] == ".pdb":
                name = name[:-4]
            names.append(name)

            with torch.no_grad():
                # run featurize to remap R_idx and add batch dimension
                if args.verbose:
                    if "Y" in list(protein_dict):
                        atom_coords = protein_dict["Y"].cpu().numpy()
                        atom_types = list(protein_dict["Y_t"].cpu().numpy())
                        atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                        number_of_atoms_parsed = np.sum(atom_mask)
                    else:
                        print("No ligand atoms parsed")
                        number_of_atoms_parsed = 0
                        atom_types = ""
                        atom_coords = []
                    if number_of_atoms_parsed == 0:
                        print("No ligand atoms parsed")
                    elif args.model_type == "ligand_mpnn":
                        print(
                            f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                        )
                        for i, atom_type in enumerate(atom_types):
                            print(
                                f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                            )
                atom_coords = protein_dict["Y"].cpu().numpy()
                atom_types = list(protein_dict["Y_t"].cpu().numpy())
                atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                number_of_atoms_parsed = np.sum(atom_mask)
                feature_dict = featurize(
                    protein_dict,
                    cutoff_for_score=float(args.ligand_mpnn_cutoff_for_score),
                    use_atom_context=0,
                    number_of_ligand_atoms=number_of_atoms_parsed,
                    model_type=args.model_type,
                )
                feature_dict["batch_size"] = args.batch_size
                B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
                # add additional keys to the feature dictionary
                feature_dict["symmetry_residues"] = remapped_symmetry_residues

                logits_list = []
                probs_list = []
                log_probs_list = []
                decoding_order_list = []
                for _ in range(1):
                    feature_dict["randn"] = torch.randn(
                        [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                        device=device,
                    )
                    if args.autoregressive_score:
                        score_dict, scores = model.score(feature_dict, use_sequence=args.use_sequence)
                    elif args.single_aa_score:
                        score_dict = model.single_aa_score(feature_dict, use_sequence=args.use_sequence)
                    else:
                        print("Set either autoregressive_score or single_aa_score to True")
                        sys.exit()
                    logits_list.append(score_dict["logits"])
                    log_probs_list.append(score_dict["log_probs"])
                    probs_list.append(torch.exp(score_dict["log_probs"]))
                    decoding_order_list.append(score_dict["decoding_order"])
                log_probs_stack = torch.cat(log_probs_list, 0)
                logits_stack = torch.cat(logits_list, 0)
                probs_stack = torch.cat(probs_list, 0)
                decoding_order_stack = torch.cat(decoding_order_list, 0)

                output_stats_path = base_folder + name + ".pt"
                out_dict = {}
                out_dict["logits"] = logits_stack.cpu().numpy()
                out_dict["probs"] = probs_stack.cpu().numpy()
                out_dict["log_probs"] = log_probs_stack.cpu().numpy()
                out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
                out_dict["native_sequence"] = feature_dict["S"][0].cpu().numpy()
                out_dict["mask"] = feature_dict["mask"][0].cpu().numpy()
                out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu().numpy() #this affects decoding order
                out_dict["seed"] = seed
                out_dict["alphabet"] = alphabet
                out_dict["residue_names"] = encoded_residue_dict_rev

                mean_probs = np.mean(out_dict["probs"], 0)
                autoreg_scores.append(scores.cpu().numpy().tolist()[0])
                std_probs = np.std(out_dict["probs"], 0)
                sequence = [restype_int_to_str[AA] for AA in out_dict["native_sequence"]]
                mean_dict = {}
                std_dict = {}
                for residue in range(L):
                    mean_dict_ = dict(zip(alphabet, mean_probs[residue]))
                    mean_dict[encoded_residue_dict_rev[residue]] = mean_dict_
                    std_dict_ = dict(zip(alphabet, std_probs[residue]))
                    std_dict[encoded_residue_dict_rev[residue]] = std_dict_

                out_dict["sequence"] = sequence
                out_dict["mean_of_probs"] = mean_dict
                out_dict["std_of_probs"] = std_dict
    
    else:
        for pdb in pdb_paths:
            if args.verbose:
                print("Scoring protein from this path:", pdb)
            # fixed_residues = fixed_residues_multi[pdb]
            # redesigned_residues = redesigned_residues_multi[pdb]
            protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                pdb,
                device=device,
                chains="",
                parse_all_atoms=0,
                parse_atoms_with_zero_occupancy=0
            )
            # make chain_letter + residue_idx + insertion_code mapping to integers
            R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
            chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
            encoded_residues = []
            for i, R_idx_item in enumerate(R_idx_list):
                tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
                encoded_residues.append(tmp)
            encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
            encoded_residue_dict_rev = dict(
                zip(list(range(len(encoded_residues))), encoded_residues)
            )


            if args.transmembrane_buried:
                buried_residues = [item for item in args.transmembrane_buried]
                buried_positions = torch.tensor(
                    [int(item in buried_residues) for item in encoded_residues],
                    device=device,
                )
            else:
                buried_positions = torch.zeros(len(encoded_residues), device=device)

            if args.transmembrane_interface:
                interface_residues = [item for item in args.transmembrane_interface]
                interface_positions = torch.tensor(
                    [int(item in interface_residues) for item in encoded_residues],
                    device=device,
                )
            else:
                interface_positions = torch.zeros(len(encoded_residues), device=device)

            protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
                1 - interface_positions
            ) + 1 * interface_positions * (1 - buried_positions)

            if args.model_type == "global_label_membrane_mpnn":
                    protein_dict["membrane_per_residue_labels"] = torch.full(
                        (len(encoded_residues),), args.global_transmembrane_label, device=device
                    )
            # if type(args.chains_to_design) == str:
            #     chains_to_design_list = args.chains_to_design.split(",")
            # else:
            chains_to_design_list = protein_dict["chain_letters"]
            chain_mask = torch.tensor(
                np.array(
                    [
                        item in chains_to_design_list
                        for item in protein_dict["chain_letters"]
                    ],
                    dtype=np.int32,
                ),
                device=device,
            )

            # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
            # if redesigned_residues:
            #     protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
            # elif fixed_residues:
            #     protein_dict["chain_mask"] = chain_mask * fixed_positions
            # else:
            protein_dict["chain_mask"] = chain_mask

            # if args.verbose:
            #     PDB_residues_to_be_redesigned = [
            #         encoded_residue_dict_rev[item]
            #         for item in range(protein_dict["chain_mask"].shape[0])
            #         if protein_dict["chain_mask"][item] == 1
            #     ]
            #     PDB_residues_to_be_fixed = [
            #         encoded_residue_dict_rev[item]
            #         for item in range(protein_dict["chain_mask"].shape[0])
            #         if protein_dict["chain_mask"][item] == 0
            #     ]
            #     print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            #     print("These residues will be fixed: ", PDB_residues_to_be_fixed)

            # # specify which residues are linked
            # if args.symmetry_residues:
            #     symmetry_residues_list_of_lists = [
            #         x.split(",") for x in args.symmetry_residues.split("|")
            #     ]
            #     remapped_symmetry_residues = []
            #     for t_list in symmetry_residues_list_of_lists:
            #         tmp_list = []
            #         for t in t_list:
            #             tmp_list.append(encoded_residue_dict[t])
            #         remapped_symmetry_residues.append(tmp_list)
            # else:
            remapped_symmetry_residues = [[]]

            # if args.homo_oligomer:
            #     if args.verbose:
            #         print("Designing HOMO-OLIGOMER")
            #     chain_letters_set = list(set(chain_letters_list))
            #     reference_chain = chain_letters_set[0]
            #     lc = len(reference_chain)
            #     residue_indices = [
            #         item[lc:] for item in encoded_residues if item[:lc] == reference_chain
            #     ]
            #     remapped_symmetry_residues = []
            #     for res in residue_indices:
            #         tmp_list = []
            #         tmp_w_list = []
            #         for chain in chain_letters_set:
            #             name = chain + res
            #             tmp_list.append(encoded_residue_dict[name])
            #             tmp_w_list.append(1 / len(chain_letters_set))
            #         remapped_symmetry_residues.append(tmp_list)

            # # set other atom bfactors to 0.0
            # if other_atoms:
            #     other_bfactors = other_atoms.getBetas()
            #     other_atoms.setBetas(other_bfactors * 0.0)

            # adjust input PDB name by dropping .pdb if it does exist
            name = pdb[pdb.rfind("/") + 1 :]
            if name[-4:] == ".pdb":
                name = name[:-4]
            names.append(name)

            with torch.no_grad():
                # run featurize to remap R_idx and add batch dimension
                if args.verbose:
                    if "Y" in list(protein_dict):
                        atom_coords = protein_dict["Y"].cpu().numpy()
                        atom_types = list(protein_dict["Y_t"].cpu().numpy())
                        atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                        number_of_atoms_parsed = np.sum(atom_mask)
                    else:
                        print("No ligand atoms parsed")
                        number_of_atoms_parsed = 0
                        atom_types = ""
                        atom_coords = []
                    if number_of_atoms_parsed == 0:
                        print("No ligand atoms parsed")
                    elif args.model_type == "ligand_mpnn":
                        print(
                            f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                        )
                        for i, atom_type in enumerate(atom_types):
                            print(
                                f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                            )
                atom_coords = protein_dict["Y"].cpu().numpy()
                atom_types = list(protein_dict["Y_t"].cpu().numpy())
                atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                number_of_atoms_parsed = np.sum(atom_mask)
                feature_dict = featurize(
                    protein_dict,
                    cutoff_for_score=float(args.ligand_mpnn_cutoff_for_score),
                    use_atom_context=0,
                    number_of_ligand_atoms=number_of_atoms_parsed,
                    model_type=args.model_type,
                )
                feature_dict["batch_size"] = args.batch_size
                B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
                # add additional keys to the feature dictionary
                feature_dict["symmetry_residues"] = remapped_symmetry_residues

                logits_list = []
                probs_list = []
                log_probs_list = []
                decoding_order_list = []
                for _ in range(1):
                    feature_dict["randn"] = torch.randn(
                        [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                        device=device,
                    )
                    if args.autoregressive_score:
                        score_dict, scores = model.score(feature_dict, use_sequence=args.use_sequence)
                    elif args.single_aa_score:
                        score_dict = model.single_aa_score(feature_dict, use_sequence=args.use_sequence)
                    else:
                        print("Set either autoregressive_score or single_aa_score to True")
                        sys.exit()
                    logits_list.append(score_dict["logits"])
                    log_probs_list.append(score_dict["log_probs"])
                    probs_list.append(torch.exp(score_dict["log_probs"]))
                    decoding_order_list.append(score_dict["decoding_order"])
                log_probs_stack = torch.cat(log_probs_list, 0)
                logits_stack = torch.cat(logits_list, 0)
                probs_stack = torch.cat(probs_list, 0)
                decoding_order_stack = torch.cat(decoding_order_list, 0)

                output_stats_path = base_folder + name + ".pt"
                out_dict = {}
                out_dict["logits"] = logits_stack.cpu().numpy()
                out_dict["probs"] = probs_stack.cpu().numpy()
                out_dict["log_probs"] = log_probs_stack.cpu().numpy()
                out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
                out_dict["native_sequence"] = feature_dict["S"][0].cpu().numpy()
                out_dict["mask"] = feature_dict["mask"][0].cpu().numpy()
                out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu().numpy() #this affects decoding order
                out_dict["seed"] = seed
                out_dict["alphabet"] = alphabet
                out_dict["residue_names"] = encoded_residue_dict_rev

                mean_probs = np.mean(out_dict["probs"], 0)
                autoreg_scores.append(scores.cpu().numpy().tolist()[0])
                std_probs = np.std(out_dict["probs"], 0)
                sequence = [restype_int_to_str[AA] for AA in out_dict["native_sequence"]]
                mean_dict = {}
                std_dict = {}
                for residue in range(L):
                    mean_dict_ = dict(zip(alphabet, mean_probs[residue]))
                    mean_dict[encoded_residue_dict_rev[residue]] = mean_dict_
                    std_dict_ = dict(zip(alphabet, std_probs[residue]))
                    std_dict[encoded_residue_dict_rev[residue]] = std_dict_

                out_dict["sequence"] = sequence
                out_dict["mean_of_probs"] = mean_dict
                out_dict["std_of_probs"] = std_dict
                # torch.save(out_dict, output_stats_path)

    out_lists = list(zip(names, autoreg_scores))
    with open(os.path.join(args.outdir, f'{args.name}_{args.scorer}_scores.csv'), 'w+') as outfile:
        outfile.write(f'Protein,{args.scorer}_Score\n')
        for entry in out_lists:
            outfile.write(str(entry[0]) + ',' + str(entry[1]) + '\n')



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--model_type",
        type=str,
        default="protein_mpnn",
        help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn",
    )
    # protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
    # ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
    # per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
    # global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
    # soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
    argparser.add_argument(
        "--checkpoint_protein_mpnn",
        type=str,
        default="./model_params/proteinmpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_ligand_mpnn",
        type=str,
        default="./model_params/ligandmpnn_v_32_010_25.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_per_residue_label_membrane_mpnn",
        type=str,
        default="./model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_global_label_membrane_mpnn",
        type=str,
        default="./model_params/global_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_soluble_mpnn",
        type=str,
        default="./model_params/solublempnn_v_48_020.pt",
        help="Path to model weights.",
    )

    argparser.add_argument("--verbose", type=int, default=1, help="Print stuff")

    argparser.add_argument(
        "--pdb_path", type=str, default="", help="Path to the input PDB."
    )
    argparser.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    argparser.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="Provide fixed residues, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--fixed_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--redesigned_residues",
        type=str,
        default="",
        help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--redesigned_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'",
    )
    
    argparser.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.",
    )

    argparser.add_argument(
        "--out_folder",
        type=str,
        help="Path to a folder to output scores, e.g. /home/out/",
    )
    argparser.add_argument(
        "--file_ending", type=str, default="", help="adding_string_to_the_end"
    )
    argparser.add_argument(
        "--zero_indexed",
        type=str,
        default=0,
        help="1 - to start output PDB numbering with 0",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set seed for torch, numpy, and python random.",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequence to generate per one pass.",
    )
    argparser.add_argument(
        "--number_of_batches",
        type=int,
        default=1,
        help="Number of times to design sequence using a chosen batch size.",
    )

    argparser.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="1 - use atom context, 0 - do not use atom context.",
    )

    argparser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="Flag to use side chain atoms as ligand context for the fixed residues",
    )

    argparser.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )

    argparser.add_argument(
        "--chains_to_design",
        type=str,
        default=None,
        help="Specify which chains to redesign, all others will be kept fixed.",
    )

    argparser.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="Provide chains letters for parsing backbones, 'ABCF'",
    )

    argparser.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    argparser.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )

    argparser.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",
    )

    argparser.add_argument(
        "--use_sequence",
        type=int,
        default=1,
        help="1 - get scores using amino acid sequence info; 0 - get scores using backbone info only",
    )

    argparser.add_argument(
        "--autoregressive_score",
        type=int,
        default=1,
        help="1 - run autoregressive scoring function; p(AA_1|backbone); p(AA_2|backbone, AA_1) etc, 0 - False",
    )

    argparser.add_argument(
        "--single_aa_score",
        type=int,
        default=0,
        help="1 - run single amino acid scoring function; p(AA_i|backbone, AA_{all except ith one}), 0 - False",
    )

    args = argparser.parse_args()
    main(args)
