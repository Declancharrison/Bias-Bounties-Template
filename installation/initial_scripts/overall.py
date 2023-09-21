#!/usr/bin/python3

import helper_functions as hf

def main():
    # reset comment file
    hf.initialize_comment()

    # request.yaml only changed file
    hf.test_len()

    # pingable URLS
    hf.test_urls()

    # run security
    flag = hf.security_call()

    # test updates
    hf.test_update(flag)

if __name__ == "__main__":
    main()