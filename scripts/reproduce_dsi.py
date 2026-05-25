import argparse
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import torch
from flyvis import results_dir
from flyvis.network import NetworkView
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis.moving_bar_responses import direction_selectivity_index, peak_responses

def main():
    parser = argparse.ArgumentParser(description="Reproduction Audit for DSI")
    parser.add_argument("--ensemble", type=str, default="0000", help="Ensemble ID to evaluate")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of networks to process")
    parser.add_argument("--angles", type=int, default=12, choices=[4, 12],
                        help="Number of angles: 12 (paper, default) or 4 (cardinal only)")
    args = parser.parse_args()

    ensemble_id = args.ensemble
    limit = args.limit
    n_angles_mode = args.angles

    # Set up angles based on mode
    if n_angles_mode == 4:
        angles_list = [0, 90, 180, 270]
    else:
        angles_list = list(np.arange(0, 360, 30))

    print(f"Starting Reproduction Audit for Ensemble {ensemble_id} (Limit: {limit}, Angles: {n_angles_mode})...")

    all_results = []

    for net_id in range(limit):
        net_id_str = f"{net_id:03d}"
        model_path = results_dir / f"flow/{ensemble_id}/{net_id_str}"
        
        if not model_path.exists():
            continue

        print(f"Processing network {net_id_str} at {model_path}...")
        
        try:
            network_view = NetworkView(model_path)
            
            # Using MovingEdge (full-field edge stimuli, matching paper)
            dataset = MovingEdge(
                offsets=(-10, 11),
                intensities=[0, 1],
                speeds=[19],
                height=80,
                post_pad_mode="continue",
                t_pre=1.0,
                t_post=1.0,
                dt=1/200,
                angles=angles_list,
            )

            # Debug Checks on First Network
            if net_id == 0:
                print("--- Debug Checks ---")
                
                print("Dataset Arg DF:")
                print(dataset.arg_df)
                
                # Check Pre-rendered content
                print(f"RenderedOffsets Max: {dataset.wrap.offsets[:].max():.4f}")
                print(f"RenderedOffsets Mean: {dataset.wrap.offsets[:].mean():.4f}")
                print(f"RenderedOffsets Shape: {dataset.wrap.offsets.shape}")
                
                # Check Sequences
                if 19 in dataset.sequences:
                    seq = dataset.sequences[19]
                    print(f"Sequence[19] Shape: {seq.shape}")
                    print(f"Sequence[19] Max (ignoring NaNs): {np.nanmax(seq.cpu().numpy()):.4f}")
                else:
                    print("Speed 19 not in sequences!")

                # Check Stimulus Identity (MovingEdge has width=80 hardcoded)
                try:
                    s0 = dataset.get(angle=0, width=80, intensity=1, speed=19).cpu().numpy()
                    s180 = dataset.get(angle=180, width=80, intensity=1, speed=19).cpu().numpy()

                    s0_clean = np.nan_to_num(s0, nan=0.0)
                    s180_clean = np.nan_to_num(s180, nan=0.0)

                    diff = np.abs(s0_clean - s180_clean).max()

                    print(f"Stim0 (clean) Max: {s0_clean.max():.4f}, Mean: {s0_clean.mean():.4f}")
                    print(f"Stim180 (clean) Max: {s180_clean.max():.4f}, Mean: {s180_clean.mean():.4f}")
                    print(f"Diff Max: {diff:.4f}")

                    if diff < 1e-6:
                        print("WARNING: Stimuli identical!")
                    else:
                        print("PASSED: Stimuli distinct.")

                except Exception as e:
                    print(f"Check failed: {e}")

            # Generate Responses
            stims_and_resps = network_view.moving_edge_responses(dataset)

            # 1. Library DSI for sanity check
            dsi_library = direction_selectivity_index(stims_and_resps)

            # 2. Manual calculation with shuffle test
            peaks = peak_responses(stims_and_resps)
            peaks = peaks.set_index(sample=["angle", "width", "intensity", "speed"]).unstack("sample")
            peaks = peaks.squeeze(dim=["width", "speed"])
            if "network_id" in peaks.dims:
                peaks = peaks.squeeze(dim="network_id")

            # Get angles from data (works for any number)
            angles_rad = np.radians(peaks.coords["angle"].values)
            n_angles = len(angles_rad)
            complex_weights = np.exp(1j * angles_rad)

            # Compute normalization across BOTH intensities (matching library behavior)
            # Library uses: normalization = np.abs(view).sum(dim='angle').max(dim='intensity')
            R_int0 = np.abs(peaks.sel(intensity=0).values)
            R_int1 = np.abs(peaks.sel(intensity=1).values)
            norm_int0 = R_int0.sum(axis=1)
            norm_int1 = R_int1.sum(axis=1)
            shared_norm = np.maximum(norm_int0, norm_int1) + 1e-15

            for intensity in [0, 1]:
                p_int = peaks.sel(intensity=intensity)
                R = p_int.values  # shape: [n_neurons, n_angles]

                # Vector sum DSI using complex exponentials
                vector_sum = R @ complex_weights
                ds_mag = np.abs(vector_sum) / shared_norm
                theta_pref = np.angle(vector_sum)

                # Shuffle test using same normalization as reported DSI
                n_shuffles = 1000
                n_neurons = R.shape[0]
                null_ds_mags = np.zeros((n_shuffles, n_neurons))

                for i in range(n_shuffles):
                    rand_indices = np.argsort(np.random.rand(n_neurons, n_angles), axis=1)
                    row_indices = np.arange(n_neurons)[:, None]
                    R_shuff = R[row_indices, rand_indices]

                    vec_sum_shuff = R_shuff @ complex_weights
                    null_ds_mags[i, :] = np.abs(vec_sum_shuff) / shared_norm

                mean_null = null_ds_mags.mean(axis=0)
                std_null = null_ds_mags.std(axis=0)
                z_score = (ds_mag - mean_null) / (std_null + 1e-15)

                # Get library values for comparison
                dsi_lib_vals = dsi_library.sel(intensity=intensity).values
                cell_types = p_int.coords["cell_type"].values

                for idx in range(n_neurons):
                    all_results.append({
                        "network_id": net_id_str,
                        "neuron_idx": idx,
                        "cell_type": cell_types[idx],
                        "intensity": intensity,
                        "dsi_library": dsi_lib_vals[idx],
                        "dsi_manual": ds_mag[idx],
                        "z_score": z_score[idx],
                        "theta_pref": theta_pref[idx],
                    })
                    
        except Exception as e:
            print(f"Error on {net_id_str}: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(all_results)
    df.to_csv(f"reproduce_dsi_results_{ensemble_id}.csv", index=False)

    # Summary
    if not df.empty:
        print("\n--- Summary Report ---")

        # Sanity check: library vs manual DSI consistency
        df["dsi_diff"] = np.abs(df["dsi_library"] - df["dsi_manual"])
        print(f"\nSanity Check (Library vs Manual DSI):")
        print(f"  Max diff: {df['dsi_diff'].max():.6f}")
        print(f"  Mean diff: {df['dsi_diff'].mean():.6f}")
        if df["dsi_diff"].max() > 0.01:
            print("  WARNING: Library and manual DSI differ significantly!")
        else:
            print("  PASSED: Library and manual DSI are consistent.")

        target_types = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
        for ct in target_types:
            inte = 1 if "T4" in ct else 0
            subset = df[(df["cell_type"] == ct) & (df["intensity"] == inte)]
            print(f"\nType: {ct} (Intensity {inte})")
            if len(subset) == 0:
                print("  N: 0")
            else:
                print(f"  N: {len(subset)}")
                print(f"  Mean DSI (library): {subset['dsi_library'].mean():.4f}")
                print(f"  Mean DSI (manual): {subset['dsi_manual'].mean():.4f}")
                print(f"  Mean Z-Score: {subset['z_score'].mean():.4f}")
                print(f"  % Sig (z>1.96): {(subset['z_score'] > 1.96).mean()*100:.1f}%")

if __name__ == "__main__":
    main()
