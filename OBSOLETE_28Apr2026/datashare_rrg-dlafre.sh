#!/bin/bash

# Script to share files with specific users
# Creates per-user folders in /scratch/eartigau/datashare

BASEDIR="/scratch/eartigau/datashare"

# Get list of users in rrg-dlafre group
GROUP_MEMBERS=($(getent group rrg-dlafre | cut -d: -f4 | tr ',' '\n' | sort))

# Display numbered list
echo "Members of rrg-dlafre group:"
echo ""
for i in "${!GROUP_MEMBERS[@]}"; do
    printf "  %2d) %s\n" $((i+1)) "${GROUP_MEMBERS[$i]}"
done
echo ""
echo "   0) Quit"
echo ""

# Prompt for selection
read -p "Select user number: " SELECTION

# Handle quit
if [[ "$SELECTION" == "0" ]]; then
    echo "Aborted."
    exit 0
fi

# Validate selection
if ! [[ "$SELECTION" =~ ^[0-9]+$ ]] || [[ "$SELECTION" -lt 1 ]] || [[ "$SELECTION" -gt ${#GROUP_MEMBERS[@]} ]]; then
    echo "Error: Invalid selection"
    exit 1
fi

INPUT_USER="${GROUP_MEMBERS[$((SELECTION-1))]}"

# Validate user exists on scratch
if [[ ! -d "/scratch/${INPUT_USER}" ]]; then
    echo "Error: User '${INPUT_USER}' does not have a /scratch/${INPUT_USER} folder"
    # wqexit 1
fi

echo "User '${INPUT_USER}' selected."

# Prompt for files/folders
read -p "Enter files or pattern to share (e.g., /path/to/file.fits or OBJECT/*.fits): " FILES

# Expand the pattern
EXPANDED_FILES=( $FILES )

# Check if any files match
if [[ ${#EXPANDED_FILES[@]} -eq 0 ]] || [[ ! -e "${EXPANDED_FILES[0]}" ]]; then
    echo "Error: No files found matching '${FILES}'"
    exit 1
fi

echo "Found ${#EXPANDED_FILES[@]} file(s) to copy."

# Create user directory if needed
USERDIR="${BASEDIR}/${INPUT_USER}"
if [[ ! -d "$USERDIR" ]]; then
    echo "Creating directory: ${USERDIR}"
    mkdir -p "$USERDIR"
fi

# Copy files, preserving folder structure if applicable
for filepath in "${EXPANDED_FILES[@]}"; do
    if [[ -f "$filepath" ]]; then
        # Get the parent directory name (for structure preservation)
        parent=$(dirname "$filepath")
        parentname=$(basename "$parent")
        
        # Check if this looks like a subfolder pattern (not just current dir)
        if [[ "$parent" != "." && "$parent" != "$PWD" && "$FILES" == *"/"* ]]; then
            # Preserve folder structure
            destdir="${USERDIR}/${parentname}"
            mkdir -p "$destdir"
            cp -vL "$filepath" "$destdir/"
        else
            # Flat copy
            cp -vL "$filepath" "$USERDIR/"
        fi
    elif [[ -d "$filepath" ]]; then
        # It's a directory, copy recursively (following symlinks)
        cp -rvL "$filepath" "$USERDIR/"
    fi
done

# Set permissions: owner (eartigau) and the specific user only
echo "Setting permissions..."

# Change group to rrg-dlafre
# chgrp -R rrg-dlafre "$USERDIR"

# Set directory permissions: rwx for owner, rx for group, nothing for others
#find "$USERDIR" -type d -exec chmod 750 {} \;

setfacl -m u:"$INPUT_USER":rx  "$BASEDIR/"
setfacl -R -m u:"$INPUT_USER":rx  "$USERDIR/"


# Set file permissions: rw for owner, r for group, nothing for others
# find "$USERDIR" -type f -exec chmod 640 {} \;

echo ""
echo "Done! Files shared to: ${USERDIR}"
echo "Permissions set for eartigau (owner) and rrg-rdoyon group."
echo ""
echo "============================================================"
echo "EMAIL TEXT FOR ${INPUT_USER}:"
echo "============================================================"
echo ""
echo "Hi,"
echo ""
echo "I've shared some files with you on fir@alliance. Here's the summary:"
echo ""
echo "Location: ${USERDIR}"
echo ""
# Count files and folders
NFILES=$(find "$USERDIR" -type f | wc -l)
NDIRS=$(find "$USERDIR" -type d | wc -l)
NDIRS=$((NDIRS - 1))  # exclude the base directory itself
echo "Contents: ${NFILES} file(s) in ${NDIRS} subfolder(s)"
echo "Total size: $(du -sh "$USERDIR" | cut -f1)"
echo ""
echo "To download everything to your laptop, use:"
echo ""
echo "  rsync -avz -e 'ssh -oport=22' ${INPUT_USER}@fir.computecanada.ca:${USERDIR}/ data_from_apero/"
echo ""
echo ""
echo "Cheers,"
echo "Étienne"
echo ""
echo "============================================================"