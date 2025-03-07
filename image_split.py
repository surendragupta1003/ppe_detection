import splitfolders

input_folder = r'C:\Users\suren\Downloads\project-4-at-2025-03-07-05-43-94053696'
output_folder = r'C:\Users\suren\Downloads\project-4-at-2025-03-07-05-43-94053696\output'

splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=1337,
    ratio=(.8, .1, .1),
    group_prefix=None,
    move=False
)
