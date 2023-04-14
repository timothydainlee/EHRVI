import argparse

def arguments(argv):
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument("--data_path",
                        type=str,
                        default="../data/snuh/log/data_diabetes_clean.csv")
    parser.add_argument("--seed",
                        type=int,
                        default=42)
    parser.add_argument("--tensorboard_path",
                        type=str,
                        default="runs/run")

    # learning hyperparameters
    parser.add_argument("--cv",
                        type=int,
                        default=10)
    parser.add_argument("--epoch",
                        type=int,
                        default=1000)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=3e-4)
    parser.add_argument("--batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--beta_shape",
                        type=float,
                        default=1/128)
    parser.add_argument("--beta_pos",
                        type=float,
                        default=1000)
    parser.add_argument("--beta_max",
                        type=float,
                        default=1/8)

    # model hyperparameters
    parser.add_argument("--hidden_size_c",
                        type=int,
                        default=256)
    parser.add_argument("--hidden_size_b",
                        type=int,
                        default=128)
    parser.add_argument("--hidden_size_d",
                        type=int,
                        default=128)
    parser.add_argument("--latent_size",
                        type=int,
                        default=32)
    parser.add_argument("--num_layers",
                        type=int,
                        default=1)

    args = parser.parse_args(argv[1:])

    args.b_cols = ["gender",
                   "or_metformin",
                   "or_SU",
                   "or_DPP4i",
                   "or_TZD",
                   "or_SGLT2i",
                   "or_aGi",
                   "or_meglitinide",
                   "inj_GLP1a",
                   "inj_basal",
                   "inj_bolus",
                   "inj_premixed",
                   "MSII",
                   "htn_ACEi_ARB",
                   "htn_b_blockers",
                   "htn_a_blockers",
                   "htn_ca_channel_blockers",
                   "htn_diuretics",
                   "htn_statin",
                   "htn_ezetimibe",
                   "htn_fibrate_omega3"]
    args.c_cols = ["glucose",
                   "HbA1c",
                   "FBS",
                   "PP2",
                   "HOMA2_pB",
                   "HOMA2_pS",
                   "HOMA2_IR",
                   "ALT_24hr",
                   "AST_24hr",
                   "ALP",
                   "GGT_24hr",
                   "bilirubin",
                   "bilirubin_24hr",
                   "albumin",
                   "eGFR",
                   "creatinine",
                   "urine_creatinine",
                   "urine_microalbumin_random",
                   "urine_ACR_random",
                   "urine_protein_random",
                   "urine_PCR_random",
                   "cholesterol",
                   "HDL_24hr",
                   "LDL_24hr",
                   "TG_24hr",
                   "SBP",
                   "DBP",
                   "age",
                   "body_height",
                   "body_weight",
                   "BMI",
                   "calcium",
                   "phosphorus",
                   "potassium",
                   "sodium",
                   "chloride",
                   "RBC_24hr",
                   "MCV",
                   "MCH",
                   "MCHC",
                   "hematocrit",
                   "PCT",
                   "PLT_24hr",
                   "WBC_24hr",
                   "lymphocyte",
                   "monocyte",
                   "segmented_neutrophil",
                   "ANC",
                   "basophil",
                   "eosinophil",
                   "protein_24hr",
                   "CO2_24hr",
                   "CK_CPK_24hr",
                   "PT_INR",
                   "aPTT_24hr",
                   "fibrinogen_24hr",
                   "TSH",
                   "free_T4"]
    return args

