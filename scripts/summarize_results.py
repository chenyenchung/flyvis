import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Summarize Reproduction Results")
    parser.add_argument("csv_file", type=str, help="Path to the results CSV file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File {args.csv_file} not found.")
        return

    print(f"\n--- Summary Report for {args.csv_file} ---")
    target_types = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
    
    for ct in target_types:
        # Filter for appropriate intensity (T4->ON=1, T5->OFF=0)
        inte = 1 if "T4" in ct else 0
        subset = df[(df["cell_type"] == ct) & (df["intensity"] == inte)]
        
        print(f"\nType: {ct} (Intensity {inte})")
        if len(subset) == 0:
            print("  N: 0")
            continue

        print(f"  N: {len(subset)}")
        print(f"  Mean Original DSI: {subset['dsi_original'].mean():.4f}")
        print(f"  Mean DS Mag:       {subset['ds_mag'].mean():.4f}")
        print(f"  Mean Z-Score:      {subset['z_score'].mean():.4f}")
        print(f"  % Significant (Z>1.96): {(subset['z_score'] > 1.96).mean()*100:.1f}%")

    print("\nDone.")

if __name__ == "__main__":
    main()

