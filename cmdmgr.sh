#!/usr/bin/bash

build() {
    make create_environment
    pipenv run
    make requirements
}

data_import() {
    if ["$VIRTUAL_ENV" == ""]; then
        echo "Virtual Environment not set up or activated"
        return
    fi

    flag=""
    filepath=""

    # Iterate over command-line arguments
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            -l|--local)   # Check for -l or --local flag
                flag="-l"
                ;;
            -e|--external) # Check for -e or --external flag
                flag="-e"
                ;;
            *)  # If an argument is not a flag, assume it's the filepath
                if [[ -z "$filepath" ]]; then
                    filepath="$1"
                fi
                ;;
        esac
        shift
    done

    # Validate that either flag is provided
    if [[ -z "$flag" ]]; then
        echo "Error: Missing required flag (-l|--local or -e|--external)."
        return
    fi

    # Validate that a filepath is provided
    if [[ -z "$filepath" ]]; then
        echo "Error: Missing filepath."
        return
    fi

    # If -l|--local flag is used, check if the file exists
    if [[ "$flag" == "-l" ]] && [[ ! -f "$filepath" ]]; then
        echo "Error: Local file does not exist at '$filepath'."
        return
    fi

    make data_import FLAG_ARG=$flag FILEPATH_ARG=$filepath
}