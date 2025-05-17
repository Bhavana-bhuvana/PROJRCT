import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class DataCleaner:

    def __init__(self, df: pd.DataFrame, verbose=True):
        self.original_df = df
        self.df = df.copy()
        self.verbose = verbose
        self.logs = []
        self.histograms = {}
        self.correlation_matrix = None
        self.heatmap = None
        self.report = ""

    def is_id_or_name(self, col):
        col_lower = col.lower()
        return any(keyword in col_lower for keyword in ['id', 'name', 'identifier'])

    def run_pipeline(self):
        self._drop_duplicates()
        self._handle_missing_values()
        self.encode_features()
        self._generate_histograms()
        self._compute_correlation()
        self._save_report()
        return self.df, self.histograms, self.heatmap, self.report

    def _log(self, message):
        if self.verbose:
            print(message)
        self.logs.append(message)

    def _save_report(self):
        self.report = "\n".join(self.logs)
        self._log("Cleaning report ready to view and download.")

    def _drop_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        self._log(f"Dropped {before - after} duplicate rows.")

    def _handle_missing_values(self):
        missing = self.df.isnull().sum().sum()
        self.df.fillna(self.df.mode().iloc[0], inplace=True)
        self._log(f"Filled {missing} missing values using mode.")

    def encode_features(self):
        self.encoded_columns = []
        self._log("Encoding categorical features (ordinal for <8 categories)...")
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cat_cols = [col for col in cat_cols if not self.is_id_or_name(col)]

        if not cat_cols:
            self._log("No categorical features to encode.")
            return

        ordinal_cols = [col for col in cat_cols if self.df[col].nunique() <= 8]
        onehot_cols = [col for col in cat_cols if col not in ordinal_cols]

        if ordinal_cols:
            self.ordinal_encoder = OrdinalEncoder()
            self.df[ordinal_cols] = self.ordinal_encoder.fit_transform(self.df[ordinal_cols])
            self.encoded_columns.extend(ordinal_cols)
            self._log(f"Ordinal encoded: {ordinal_cols}")

        if onehot_cols:
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            onehot_encoded = self.onehot_encoder.fit_transform(self.df[onehot_cols])
            encoded_df = pd.DataFrame(onehot_encoded,
                                      columns=self.onehot_encoder.get_feature_names_out(onehot_cols),
                                      index=self.df.index)
            self.df.drop(columns=onehot_cols, inplace=True)
            self.df = pd.concat([self.df, encoded_df], axis=1)
            self.encoded_columns.extend(encoded_df.columns.tolist())
            self._log(f"One-hot encoded: {onehot_cols} into {list(encoded_df.columns)}")

    def _generate_histograms(self):
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            self.df[col].hist(ax=ax, bins=20)
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            self.histograms[col] = buf
            plt.close(fig)
            self._log(f"Histogram generated for {col}")
            


    def _compute_correlation(self):
        numeric_df = self.df.select_dtypes(include='number')
        if numeric_df.shape[1] < 2:
            self._log("Not enough numeric columns to compute correlation.")
            return
        corr_matrix = numeric_df.corr().round(2)
        corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))  # Remove self-correlation
        self.correlation_matrix = corr_matrix

        # Save heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        self.heatmap = buf
        plt.close(fig)
        self._log("Correlation heatmap generated.")
