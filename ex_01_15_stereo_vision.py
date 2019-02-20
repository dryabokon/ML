import cv2
import numpy
import os
# ---------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_alg_match
import tools_image
import tools_draw_numpy
# ---------------------------------------------------------------------------------------------------------------------

def demo_stereo_01_SGBM():
	folder_input = 'data/ex15/'
	folder_output = 'data/output/'

	filenameL = '1L.png'
	filenameR = '1R.png'
	disp_v1, disp_v2, disp_h1, disp_h2 = 0, 1, -70, -10

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	imgR = cv2.imread(folder_input + filenameL, 0)
	imgL = cv2.imread(folder_input + filenameR, 0)

	dispL, dispR = tools_alg_match.get_disparity_SGBM(imgL, imgR, disp_v1, disp_v2, disp_h1, disp_h2)
	cv2.imwrite(folder_output + 'IL_disp.png', tools_alg_match.visualize_matches_map(dispR, disp_v1, disp_v2, disp_h1, disp_h2))
	cv2.imwrite(folder_output + 'IR_disp.png', tools_alg_match.visualize_matches_map(dispL, disp_v1, disp_v2,  disp_h1,  disp_h2))


	return
# ---------------------------------------------------------------------------------------------------------------------
def demo_stereo_02_BM():
	folder_input = 'data/ex15/'
	folder_output = 'data/output/'

	filenameL = '1L.png'
	filenameR = '1R.png'
	disp_v1, disp_v2, disp_h1, disp_h2 = 0, 1, -70, -10

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	imgL = cv2.imread(folder_input + filenameL, 0)
	imgR = cv2.imread(folder_input + filenameR, 0)
	dispL, dispR = tools_alg_match.get_disparity_BM(imgL, imgR, disp_v1, disp_v2, disp_h1, disp_h2)
	cv2.imwrite(folder_output + 'IL_disp.png', tools_alg_match.visualize_matches_map(dispL, disp_v1, disp_v2,  disp_h1,  disp_h2))
	cv2.imwrite(folder_output + 'IR_disp.png', tools_alg_match.visualize_matches_map(dispR, disp_v1, disp_v2,  disp_h1,  disp_h2))


	return
# ---------------------------------------------------------------------------------------------------------------------
def demo_stereo_03_keypoints():
	folder_input = 'data/ex15/'
	folder_output = 'data/output/'

	filenameL = '1L.png'
	filenameR = '1R.png'
	disp_v1, disp_v2, disp_h1, disp_h2 = 0, 1, -70, -10

	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	imgL = cv2.imread(folder_input + filenameL)
	imgR = cv2.imread(folder_input + filenameR)
	imgL_gray_rgb = tools_image.desaturate(imgL)
	imgR_gray_rgb = tools_image.desaturate(imgR)

	points1, des1 = tools_alg_match.get_keypoints_desc(imgL)
	points2, des2 = tools_alg_match.get_keypoints_desc(imgR)

	match1, match2, distance = tools_alg_match.get_matches_from_keypoints_desc(points1, des1, points2, des2)

	idx=[]

	for i in range(0, match1.shape[0]):
		row1, col1 = match1[i, 1], match1[i, 0]
		row2, col2 = match2[i, 1], match2[i, 0]
		if (col2 - col1>=disp_h1) and (col2 - col1<disp_h2) and (row2 - row1>=disp_v1) and (row2 - row1 < disp_v2):
			idx.append(i)

	match1, match2, distance = match1[idx], match2[idx], distance[idx]

	for i in range(0, match1.shape[0]):
		r = int(255 * numpy.random.rand())
		color = cv2.cvtColor(numpy.array([r, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)
		imgL_gray_rgb = tools_draw_numpy.draw_circle(imgL_gray_rgb, match1[i, 1], match1[i, 0], 4, color)
		imgR_gray_rgb = tools_draw_numpy.draw_circle(imgR_gray_rgb, match2[i, 1], match2[i, 0], 4, color)


	cv2.imwrite(folder_output + 'matches03_L.png', imgL_gray_rgb)
	cv2.imwrite(folder_output + 'matches03_R.png', imgR_gray_rgb)

	return
# ---------------------------------------------------------------------------------------------------------------------
def demo_stereo_04_keypoints_limited():
	folder_input = 'data/ex15/'
	folder_output = 'data/output/'

	filenameL = '1L.png'
	filenameR = '1R.png'
	disp_v1, disp_v2, disp_h1, disp_h2 = 0, 1, -70, -10


	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	imgL = cv2.imread(folder_input + filenameL)
	imgR = cv2.imread(folder_input + filenameR)
	imgL_gray_rgb = tools_image.desaturate(imgL)
	imgR_gray_rgb = tools_image.desaturate(imgR)

	points1, des1 = tools_alg_match.get_keypoints_desc(imgL)
	points2, des2 = tools_alg_match.get_keypoints_desc(imgR)

	match1, match2, distance = tools_alg_match.get_matches_from_desc_limit_by_disp(points1, des1, points2, des2,disp_v1, disp_v2, disp_h1, disp_h2,'ccc')

	R=4

	for i in range(0, match1.shape[0]):
		r = int(255 * numpy.random.rand())
		color = cv2.cvtColor(numpy.array([r, 255, 225], dtype=numpy.uint8).reshape(1, 1, 3), cv2.COLOR_HSV2BGR)
		imgL_gray_rgb = tools_draw_numpy.draw_circle(imgL_gray_rgb, match1[i, 1], match1[i, 0], R, color)
		imgR_gray_rgb = tools_draw_numpy.draw_circle(imgR_gray_rgb, match2[i, 1], match2[i, 0], R, color)

	cv2.imwrite(folder_output + 'matches03_L.png', imgL_gray_rgb)
	cv2.imwrite(folder_output + 'matches03_R.png', imgR_gray_rgb)

	return
# ---------------------------------------------------------------------------------------------------------------------
def demo_stereo_05_matchTemplate():
	folder_input = 'data/ex15/'
	folder_output = 'data/output/'

	filenameL = '1L.png'
	filenameR = '1R.png'
	disp_v1, disp_v2, disp_h1, disp_h2 = -2, +2, -70, -10


	if not os.path.exists(folder_output):
		os.makedirs(folder_output)
	else:
		tools_IO.remove_files(folder_output)

	imgL = cv2.imread(folder_input + filenameL)
	imgR = cv2.imread(folder_input + filenameR)
	imgL_gray_rgb = tools_image.desaturate(imgL)
	imgR_gray_rgb = tools_image.desaturate(imgR)


	coord1, coord2,quality = tools_alg_match.get_best_matches(imgL,imgR,disp_v1, disp_v2, disp_h1, disp_h2,window_size=15,step=5)

	dispL, dispR = tools_alg_match.get_disparity_from_matches(imgL.shape[0], imgL.shape[1], coord1, coord2,disp_v1, disp_v2, disp_h1, disp_h2)
	cv2.imwrite(folder_output + 'L_disp.png', dispL)
	cv2.imwrite(folder_output + 'R_disp.png', dispR)

	return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


	demo_stereo_01_SGBM()
	demo_stereo_02_BM()
	demo_stereo_03_keypoints()
	demo_stereo_04_keypoints_limited()
	demo_stereo_05_matchTemplate()
