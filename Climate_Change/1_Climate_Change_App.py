import numpy as np
import pandas as pd
import streamlit as st
import polars as pl
from functime.cross_validation import train_test_split
from functime.feature_extraction import add_fourier_terms
from functime.forecasting import linear_model
from functime.preprocessing import scale
from functime.metrics import mase

st.write("# Welcome to the Climate Change Dashboard! 👋")

st.markdown("""
        **👈 Select a page from the dropdown on the left** to see some applications of the app!"""
        )
