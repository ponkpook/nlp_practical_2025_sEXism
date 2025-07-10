import pandas as pd
import os

def load_augmented_data(
    include_backtranslation: bool = False,
    include_translation: bool = False,
    include_aeda: bool = False ) -> pd.DataFrame:

    ORIGINAL_FILEPATH = 'data/original.csv'
    BACKTRANSLATION_FILEPATH = 'data/augmented/backtranslation.csv'
    TRANSLATION_FILEPATH = 'data/augmented/translation.csv'
    AEDA_FILEPATH = 'data/augmented/aeda.csv'
    # -------------------------------------------------------------------

    all_dfs = []

    try:
        df_original = pd.read_csv(ORIGINAL_FILEPATH)
        all_dfs.append(df_original)
        print(f"Loaded original data: {len(df_original)} rows.")
    except FileNotFoundError:
        print(f"Error: Original data file not found at '{ORIGINAL_FILEPATH}'. Cannot proceed without original data.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading original data from '{ORIGINAL_FILEPATH}': {e}")
        return pd.DataFrame()

    
    if include_backtranslation:
        try:
            df_bt = pd.read_csv(BACKTRANSLATION_FILEPATH)
            all_dfs.append(df_bt)
            print(f"Loaded 'backtranslation' data: {len(df_bt)} rows.")
        except FileNotFoundError:
            print(f"Warning: Backtranslation file not found at '{BACKTRANSLATION_FILEPATH}'. Skipping.")
        except Exception as e:
            print(f"Error loading backtranslation data from '{BACKTRANSLATION_FILEPATH}': {e}. Skipping.")
    else:
        print("Backtranslation not included.")

    if include_translation:
        try:
            df_trans = pd.read_csv(TRANSLATION_FILEPATH)
            all_dfs.append(df_trans)
            print(f"Loaded 'translation' data: {len(df_trans)} rows.")
        except FileNotFoundError:
            print(f"Warning: Translation file not found at '{TRANSLATION_FILEPATH}'. Skipping.")
        except Exception as e:
            print(f"Error loading translation data from '{TRANSLATION_FILEPATH}': {e}. Skipping.")
    else:
        print("Translation not included.")

    if include_aeda:
        try:
            df_aeda = pd.read_csv(AEDA_FILEPATH)
            all_dfs.append(df_aeda)
            print(f"Loaded 'aeda' data: {len(df_aeda)} rows.")
        except FileNotFoundError:
            print(f"Warning: AEDA file not found at '{AEDA_FILEPATH}'. Skipping.")
        except Exception as e:
            print(f"Error loading AEDA data from '{AEDA_FILEPATH}': {e}. Skipping.")
    else:
        print("AEDA not included.")

    # 3. Concatenate all DataFrames
    if not all_dfs:
        print("No data was successfully loaded. Returning an empty DataFrame.")
        return pd.DataFrame() # In case even original failed

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully combined {len(all_dfs)} datasets into one with {len(combined_df)} rows.")

    return combined_df