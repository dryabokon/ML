# ----------------------------------------------------------------------------------------------------------------------
import tools_anomaly
step = 12
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    path_out = 'data/output/'
    filename_in = 'example.png'

    path_in, pattern_height, pattern_width, max_period, window_size = 'data/ex14/ex02/', 180, 147, 6, 35
    #path_in, pattern_height, pattern_width, max_period, window_size = 'data/ex14/ex03/', 280, 200, 5, 55
    #path_in, pattern_height, pattern_width, max_period, window_size = 'data/ex14/ex04/', 230, 170, 4, 40
    #path_in, pattern_height, pattern_width, max_period, window_size = 'data/ex14/ex06/',  75, 160, 2, 20
    #path_in, pattern_height, pattern_width, max_period, window_size = 'data/ex14/ex08/', 375, 260, 5, 85

    tools_anomaly.E2E_detect_patterns(path_in, filename_in, path_out, pattern_height, pattern_width, max_period, window_size, step)

