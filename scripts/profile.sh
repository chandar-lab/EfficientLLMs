

# for profiling
nsys profile -w true -t cuda -s cpu --capture-range=cudaProfilerApi --capture-range-end stop -x true -f true -o toy_example python nsight_profile.py
# for creating report
nsys stats --output report_toy_example --report gpukernsum --force-overwrite true toy_example.nsys-rep
nsys stats --output report_toy_example --report cudaapisum --force-overwrite true toy_example.nsys-rep