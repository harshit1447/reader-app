def auth_page():
    st.markdown("<h1 style='text-align:center'>Welcome to Reader</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 0.6, 1])
    with col2:
        st.subheader("Login or create an account")
        mode = st.radio("Select", ["Login", "Sign up"], horizontal=True)

        def safe_rerun():
            # Try to rerun the app; if experimental_rerun is not available or raises,
            # fall back to st.stop() which ends this run and allows the next run
            try:
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                else:
                    st.stop()
            except Exception:
                # If rerun fails, stop execution here so Streamlit can re-render next time.
                st.stop()

        if mode == "Sign up":
            st.write("Create a new account")
            su_display = st.text_input("Display name (optional)", key="su_disp")
            su_user = st.text_input("Username", key="su_user")
            su_email = st.text_input("Email (optional)", key="su_email")
            su_pass = st.text_input("Password", type="password", key="su_pass")
            if st.button("Create account"):
                ok, msg = create_user(su_user, su_pass, display_name=su_display or None, email=su_email or None)
                if ok:
                    uid = verify_user(su_user, su_pass)
                    st.session_state.user_id = int(uid) if uid else None
                    st.session_state.page = "app"
                    st.success("Account created and signed in — welcome!")
                    safe_rerun()
                else:
                    st.error(msg)

        else:
            st.write("Sign in to your account")
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Log in"):
                uid = verify_user(li_user, li_pass)
                if uid:
                    st.session_state.user_id = int(uid)
                    st.session_state.page = "app"
                    st.success("Signed in")
                    safe_rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown("---")
        st.write("Or continue as guest (history won't be saved).")
        if st.button("Continue as guest"):
            st.session_state.user_id = None
            st.session_state.page = "app"
            safe_rerun()
