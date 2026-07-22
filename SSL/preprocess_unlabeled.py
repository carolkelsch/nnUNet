"""
Preprocess UNLABELED images using the exact same nnU-Net plans/dataset.json
that were generated for your labeled dataset, so the resulting arrays are
directly compatible with nnUNetDataset / nnUNetDataLoader.

Run this AFTER `nnUNetv2_plan_and_preprocess` has already been run on your
LABELED dataset (so nnUNet_preprocessed/<DatasetXXX_Name>/nnUNetPlans.json
and dataset.json already exist).

IMPORTANT: nnU-Net v2's internal preprocessing API (DefaultPreprocessor,
run_case_save signature) has shifted slightly across releases. Before
running this, open:
    nnunetv2/preprocessing/preprocessors/default_preprocessor.py
in your installed version and confirm the method name/argument order below
still matches. If it doesn't, adjust the call accordingly -- the important
part conceptually is: seg_file=None triggers the same code path nnU-Net
already uses to preprocess images at inference time (nnUNetv2_predict),
which internally creates an all-zero placeholder segmentation. That's what
lets us reuse the standard nnUNetDataset/DataLoader machinery unmodified
for the unlabeled cases.
"""
import os
import argparse

from batchgenerators.utilities.file_and_folder_operations import load_json, join, maybe_mkdir_p, load_pickle, write_pickle

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor


def preprocess_unlabeled_cases(
    dataset_name: str,
    unlabeled_images_folder: str,
    output_folder: str,
    plans_identifier: str = "nnUNetPlans",
    configuration: str = "3d_fullres",
    file_ending: str = ".nii.gz",
    num_input_channels: int = 1,
):
    """
    dataset_name:            e.g. "<DatasetXXX_Name>" (must match your labeled dataset)
    plans_identifier:        e.g. "nnUNetPlans"
    configuration:           e.g. "3d_fullres" / "2d" / "3d_lowres"
    unlabeled_images_folder: folder with the raw unlabeled images, named like
                             imagesTr (e.g. "case_012_0000.nii.gz" for channel 0 and so on)
    output_folder:           where preprocessed .npy/.pkl files are written, e.g.
                             nnUNet_preprocessed/<DatasetXXX_Name>/imagesTrUnlabeled
    """
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    plans = load_json(join(preprocessed_dataset_folder, plans_identifier + ".json"))
    dataset_json = load_json(join(nnUNet_raw, dataset_name, "dataset.json"))

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    preprocessor = DefaultPreprocessor()

    maybe_mkdir_p(output_folder)

    all_files = sorted(f for f in os.listdir(unlabeled_images_folder) if f.endswith(file_ending))
    # strip the "_0000.nii.gz"-style channel suffix to get case identifiers
    suffix_len = len(file_ending) + 5  # "_0000" + extension
    case_ids = sorted(set(f[:-suffix_len] for f in all_files))

    print(f"Found {len(case_ids)} unlabeled cases in {unlabeled_images_folder}")

    for case_id in case_ids:
        image_files = [
            join(unlabeled_images_folder, f"{case_id}_{c:04d}{file_ending}")
            for c in range(num_input_channels)
        ]
        output_filename_truncated = join(output_folder, case_id)

        preprocessor.run_case_save(
            output_filename_truncated,
            image_files,
            None,  # seg_file=None -> zero placeholder segmentation, same as inference preprocessing
            plans_manager,
            configuration_manager,
            dataset_json,
        )

        # seg_file=None means has_seg=False during preprocessing, so 'class_locations'
        # is never added to properties -- but nnUNetDataLoader unconditionally reads
        # properties['class_locations'] at the call site in generate_train_batch(),
        # regardless of oversample_foreground_percent. Add an empty placeholder so it
        # doesn't KeyError. This is safe as long as you keep oversample_foreground_percent=0
        # for the unlabeled loader (and aren't using an ignore_label) -- the dict's
        # contents are never actually read in that case, just its presence is checked.
        pkl_path = output_filename_truncated + ".pkl"
        properties = load_pickle(pkl_path)
        properties["class_locations"] = {}
        write_pickle(properties, pkl_path)

        print(f"  preprocessed {case_id}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='PreprocessUnlabeledData',
        description='This code preprocess the unlabelled data according to the labeled processed plans from nnUNet.',
        epilog="Let's get started!"
    )
    parser.add_argument('-ds', '--dataset_name', type=str, required=True, help="Dataset name in the nnUNet format 'DatasetXXX_Name'.")
    parser.add_argument('-ch', '--input_channels', type=int, required=True, help="Number of channels from the dataset.")
    parser.add_argument('--resolution', type=str, required=False, default="3d_fullres", help="Resolution of the nnUNet plans, defaults to '3d_fullres'.")
    parser.add_argument('--plans_identifier', type=str, default="nnUNetPlans", required=False, help="Plans identifier, defaults to 'nnUNetPlans'.")
    parser.add_argument('--img_ending', type=str, default=".nii.gz", required=False, help="Image file ending")

    args = parser.parse_args()

    print(args)

    preprocess_unlabeled_cases(
        dataset_name=args.dataset_name,
        plans_identifier=args.plans_identifier,
        configuration=args.resolution,
        unlabeled_images_folder=join(nnUNet_raw, args.dataset_name, "imagesTrUnlabeled"),
        output_folder=join(nnUNet_preprocessed, args.dataset_name, "imagesTrUnlabeled"),
        file_ending=args.img_ending,
        num_input_channels=args.input_channels
    )