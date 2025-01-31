import streamlit as st
from home_screen import main as home_screen
from market_simulator import main as trading_screen

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Route to appropriate page
    if st.session_state.page == 'home':
        home_screen()
    elif st.session_state.page == 'trade':
        trading_screen()

if __name__ == "__main__":
    main()