# Week 1 (Days 1–8) — Development Environment & Core Foundations

**Focus:** Environment setup, Python refresh (data structures, OOP, async), and AI/ML libraries.  
**Deliverables:** Basic Python scripts for data structures, OOP, and asynchronous programming; Jupyter Notebook for NumPy, Pandas, Matplotlib, and Seaborn data analysis.

## Step-by-step

1) **Set up environment**
   - Created a local virtual environment using Python 3.11+.
   - Installed and configured Visual Studio Code with Python and Jupyter extensions.
   - Initialized the Git repository.

2) **Python Fundamentals Refresh**
   - **Lists, Dictionaries, Sets, and Tuples** in `src/week1/data_structures.py`.
   - **Object-Oriented Programming (OOP) concepts** (Encapsulation, Inheritance) in `src/week1/oop.py` defining the `TutorBot` class.
   - **Asynchronous programming** using `asyncio` in `src/week1/async_example.py`.

3) **AI/ML Library Practice**
   - File: `src/week1/Suyasha_Rai_Log_1.ipynb`
   - Demonstrated usage of:
     - **NumPy**: Array operations and vector arithmetic.
     - **Pandas**: DataFrame creation, filtering, and data profiling.
     - **Matplotlib/Seaborn**: Plotting statistics and visual representation of tabular data.

## Challenges and Resolutions
- **Asyncio in Jupyter:** Encountered a `RuntimeError` regarding nested asyncio event loops. Resolved by leveraging the existing loop directly in the notebook (e.g. `await main()`) rather than calling `asyncio.run()`, which is used in standalone Python scripts.
