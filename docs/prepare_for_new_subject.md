### Generate a prb file describing geometry of electrode array

1.	Place the array configuration files (particularly the .cmp file) in `/path/to/marmoset_data/array_map_files`
	- Make sure that you have swapped banks A and B in the .cmp file (see other subject folders as examples).
2.	Edit and run the following code by adding a new dict entry 
for the array. Use existing dict entries as a template: 
[/neural/generate_prbfile_from_mapfile.py](/neural/generate_prbfile_from_mapfile.py)

### Make the metadata file
Copy any one of the the metadata files located at `/path/to/marmoset_data/metadata_yml_files`
and adapt it for the current subject.

### Make subject-specific scripts
1.	Create a directory in `/subject_specific_scripts` for the new subject.
2.	Save copies of all TEMPLATE files in [/subject_specific_scripts/TEMPLATES](/subject_specific_scripts/TEMPLATES) 
to your new subject folder and rename with the subjectID replacing "TEMPLATE".
