import os
import re
import argparse
import torch
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

"""
This script uses the ProstT5 model from Rostlab to create combined amino acid (AA) 
sequence and 3Di structure embeddings for a set of transcription factors (TFs).

This is done by loading the AA sequence for each TF from a FASTA file, matching it to a
corresponding 3Di sequence from a Foldseek output FASTA file, and then using the ProstT5
tokenizer and model to create a combined embedding. The resulting embeddings are saved as
PyTorch tensors in the specified output directory.

This script is from the TFBindFormer project. See the original code and documentation here:
https://github.com/BioinfoMachineLearning/TFBindFormer

Read the original paper here:
https://www.biorxiv.org/content/10.64898/2026.04.09.717563v2
"""


def embedding_features(tokenizer, model, seq_1d, seq_3di, d_model, device):
    d1 = len(seq_1d)
    d2 = len(seq_3di)

    # preprocess sequences
    seq_1d = " ".join(list(re.sub(r"[UZOB]", "X", seq_1d)))
    seq_3di = " ".join(list(seq_3di.lower()))

    # Add special tokens to indicate sequence type for the model
    input_seqs = [
        "<AA2fold> " + seq_1d,
        "<fold2AA> " + seq_3di,
    ]

    # Tokenize and encode sequences using the ProstT5 tokenizer
    ids = tokenizer(
        input_seqs,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(
                ids.input_ids, attention_mask=ids.attention_mask
            )

    # Extract the embeddings for the AA and 3Di sequences (last hidden state)
    emb_aa = outputs.last_hidden_state[0, 1 : d1 + 1]
    emb_3di = outputs.last_hidden_state[1, 1 : d2 + 1]

    # Trim the two sequences to the same length and concatenate along the feature dimension
    L = min(d1, d2)
    emb = torch.cat(
        [emb_aa[:L], emb_3di[:L]], dim=-1
    ).float()

    # Project the concatenated embeddings to the desired output dimension using a 
    # simple feedforward network
    proj = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, d_model),
    ).to(device)

    with torch.no_grad():
        emb = proj(emb)

    return emb.cpu()


def extract_np_accession(text):
    match = re.search(r"(NP_\d+\.\d+)", text)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Extract TF embeddings using ProstT5 (AA + 3Di)"
    )
    parser.add_argument(
        "--aa_dir", required=True, help="Directory with AA FASTA files"
    )
    parser.add_argument(
        "--di_fasta", required=True, help="Foldseek 3Di FASTA file eg. ../3di_out/pdb_3Di_ss.fasta"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory"
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Output embedding dimension (default: 512)"
    )
    parser.add_argument(
        "--device", default="cuda", help="cuda or cpu"
    )

    args = parser.parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.aa_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all 3Di sequences
    logging.info(f"Loading 3Di sequences from {args.di_fasta}")
    di_dict = {}
    
    # Load the ProstT5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/ProstT5",
        revision="refs/pr/2",
        do_lower_case=False,
    )

    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5",
        revision="refs/pr/2",
        use_safetensors=True,
    ).to(device)
    
    if device.type == "cpu":
        model.float()
    else:
        model.half()
    
    # Load 3Di sequences into a dictionary keyed by accession number (e.g. NP_123456.1)
    for rec in SeqIO.parse(args.di_fasta, "fasta"):
        header = f"{rec.id} {rec.description}"
        acc = extract_np_accession(header) or rec.id.split()[0]
        di_dict[acc] = str(rec.seq)
        
    # Process the AA FASTA files and extract embeddings
    for fname in os.listdir(args.aa_dir):
        if not fname.endswith(".fasta"):
            continue
        
        # The TF files should be named like "TFNAME.fasta"
        tf_id = fname.replace(".fasta", "")
        out_path = os.path.join(
            args.out_dir, f"{tf_id}_embedding.pt"
        )
        
        if os.path.exists(out_path):
            logging.info(f"Embedding for {tf_id} already exists at {out_path}, skipping")
            continue        
        
        # Load the AA sequence for this TF
        aa_path = os.path.join(args.aa_dir, fname)
        aa_rec = next(SeqIO.parse(aa_path, "fasta"))
        
        # Extract the NP accession number for the TF so it can be matched to the 3Di sequence
        aa_header = f"{aa_rec.id} {aa_rec.description}"
        acc = extract_np_accession(aa_header) or tf_id

        if acc not in di_dict:
            logging.warning(f"No 3Di for {tf_id} (acc: {acc}), skipping")
            continue
        
        # Get the AA and matching 3Di sequence
        aa_seq = str(aa_rec.seq)
        di_seq = di_dict[acc]

        # Use the ProsT5 tokenizer and model to create combined AA + 3Di embeddings
        emb = embedding_features(tokenizer, model, aa_seq, di_seq, args.d_model, device)

        # Save the embedding to the output directory
        torch.save(emb, out_path)

        logging.info(f"Saved {tf_id}: {tuple(emb.shape)} → {out_path}")
        
    logging.info("\nAll done!")


if __name__ == "__main__":
    main()