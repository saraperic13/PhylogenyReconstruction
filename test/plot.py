import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('accuracy - blue, loss - red')
ax1.set_title('100-pair-sized batch/iter')

t = np.arange(0, 100, 1)
accuracy = [0.6, 0.69, 0.69, 0.72, 0.63, 0.71, 0.72, 0.69, 0.63, 0.76, 0.68, 0.72, 0.62, 0.67, 0.74, 0.69, 0.6, 0.73, 0.65, 0.67, 0.58, 0.65, 0.66, 0.67, 0.62, 0.66, 0.68, 0.67, 0.7, 0.67, 0.7, 0.67, 0.62, 0.57, 0.72, 0.68, 0.72, 0.63, 0.68, 0.67, 0.69, 0.69, 0.67, 0.68, 0.62, 0.64, 0.64, 0.68, 0.7, 0.7, 0.63, 0.71, 0.6, 0.73, 0.7, 0.66, 0.73, 0.67, 0.72, 0.7, 0.68, 0.71, 0.64, 0.73, 0.64, 0.7, 0.7, 0.66, 0.71, 0.53, 0.66, 0.69, 0.67, 0.6, 0.62, 0.71, 0.67, 0.73, 0.66, 0.62, 0.69, 0.67, 0.68, 0.67, 0.71, 0.65, 0.68, 0.63, 0.6, 0.68, 0.55, 0.63, 0.67, 0.61, 0.64, 0.6, 0.63, 0.63, 0.64, 0.73]
# acc2 = [0.75, 1.0, 1.0, 0.53, 0.65, 0.55, 1.0, 0.0, 0.59, 0.41, 0.52, 1.0, 0.35, 0.62, 0.61, 0.43, 0.69, 0.0, 0.7, 1.0, 0.71, 0.74, 1.0, 0.66, 0.59, 0.0, 0.71, 0.52, 0.66, 0.0, 0.5, 0.68, 0.73, 0.0, 0.49, 0.31, 0.84, 0.6, 0.72, 0.49, 0.54, 0.0, 1.0, 1.0, 0.28, 0.55, 0.6, 0.0, 0.43, 0.69, 0.68, 0.66, 0.37, 0.78, 0.75, 0.56, 0.74, 0.5, 1.0, 0.72, 0.59, 0.64, 0.68, 0.64, 0.68, 0.6, 0.8, 1.0, 0.41, 0.37, 0.76, 1.0, 0.41, 0.0, 0.78, 0.35, 0.43, 0.48, 0.63, 0.44, 0.36, 0.25, 0.71, 1.0, 0.65, 0.66, 0.82, 0.67, 0.58, 0.43, 0.0, 1.0, 0.44, 0.0, 0.83, 0.78, 1.0, 0.69, 0.54, 0.27]
# acc3 = [0.64, 0.8, 0.26, 0.83, 0.45, 0.67, 0.69, 0.4, 0.0, 0.0, 0.57, 0.0, 0.0, 0.62, 0.0, 0.75, 0.66, 0.78, 0.72, 0.69, 0.0, 0.79, 0.72, 0.74, 0.66, 0.0, 0.73, 0.34, 0.0, 0.66, 0.73, 0.34, 0.66, 0.0, 0.68, 0.72, 0.81, 0.4, 0.0, 0.0, 0.41, 0.0, 0.82, 0.43, 0.0, 0.41, 0.64, 0.7, 0.34, 0.0, 0.78, 0.71, 0.6, 0.74, 0.75, 0.35, 0.0, 0.79, 0.41, 0.81, 0.69, 0.0, 0.0, 0.62, 0.36, 0.45, 0.3, 0.76, 0.75, 0.0, 0.28, 0.64, 0.73, 0.0, 0.35, 0.66, 0.4, 0.8, 0.5, 0.78, 0.0, 0.82, 0.0, 0.0, 0.73, 0.32, 0.6, 0.66, 0.69, 0.7, 0.28, 0.74, 0.68, 0.8, 0.77, 0.4, 0.0, 0.3, 0.0, 0.78]
# acc4 = [0.67, 0.42, 0.29, 0.08, 0.25, 0.0, 0.59, 0.69, 0.0, 0.55, 0.0, 0.45, 0.0, 0.91, 0.0, 0.78, 0.32, 0.77, 0.0, 0.0, 0.75, 0.54, 0.73, 0.0, 0.66, 0.7, 0.0, 0.0, 0.0, 0.7, 0.5, 0.6, 0.0, 0.0, 0.81, 0.0, 0.27, 0.31, 0.69, 0.85, 0.0, 0.14, 0.28, 0.62, 0.81, 0.72, 0.49, 0.0, 0.0, 0.35, 0.19, 0.34, 0.66, 0.66, 0.0, 0.65, 0.51, 0.0, 0.66, 0.0, 0.0, 0.39, 0.7, 0.36, 0.66, 0.53, 0.65, 0.78, 0.63, 0.0, 0.47, 0.0, 0.0, 0.67, 0.37, 0.76, 0.38, 0.79, 0.7, 0.0, 0.67, 0.88, 0.59, 0.72, 0.64, 0.0, 0.0, 0.76, 0.71, 0.67, 0.3, 0.1, 0.64, 0.56, 0.0, 0.44, 0.8, 0.0, 0.79, 0.67]
# acc5 = [0.69, 0.45, 0.8, 0.4, 0.58, 0.42, 0.79, 0.0, 0.72, 0.35, 0.0, 0.0, 0.0, 0.0, 0.71, 0.26, 0.74, 0.61, 0.82, 0.0, 0.51, 0.84, 0.0, 0.39, 0.7, 0.49, 0.4, 0.81, 0.71, 0.77, 0.0, 0.43, 0.38, 0.27, 0.74, 0.43, 0.48, 0.71, 0.71, 0.82, 0.54, 0.63, 0.65, 0.27, 0.37, 0.0, 0.0, 0.77, 0.82, 0.85, 0.78, 0.64, 0.72, 0.37, 0.0, 0.36, 0.48, 0.34, 0.0, 0.64, 0.76, 0.73, 0.0, 0.4, 0.77, 0.8, 0.0, 0.49, 0.79, 0.76, 0.39, 0.46, 0.45, 0.64, 0.32, 0.41, 0.72, 0.68, 0.0, 0.76, 0.0, 0.25, 0.65, 0.0, 0.54, 0.33, 0.72, 0.0, 0.74, 0.52, 0.72, 0.73, 0.33, 0.65, 0.34, 0.8, 0.44, 0.65, 0.45, 0.32]

loss =[0.77001786, 0.6523317, 0.5979531, 0.6289457, 0.73690575, 0.65698224, 0.6145098, 0.66143525, 0.7962999, 0.59133124, 0.6695462, 0.6279862, 0.747433, 0.65948623, 0.5923297, 0.6834318, 0.7777503, 0.5898922, 0.6997528, 0.68826735, 0.81099856, 0.75307786, 0.7016517, 0.71366465, 0.74293184, 0.7221941, 0.70083815, 0.6931044, 0.6592546, 0.693191, 0.6677916, 0.7274142, 0.78113806, 0.8264096, 0.6529404, 0.7119021, 0.6455623, 0.75650024, 0.67772245, 0.6571269, 0.66342795, 0.7292137, 0.73792326, 0.6815715, 0.8112703, 0.7286446, 0.70941424, 0.6318478, 0.6805555, 0.6591547, 0.79116255, 0.6142866, 0.8008633, 0.58899504, 0.6467693, 0.7317725, 0.61210763, 0.6921472, 0.6289788, 0.663724, 0.7051817, 0.63979876, 0.72300094, 0.66044617, 0.7464447, 0.6272619, 0.67582923, 0.7178418, 0.63983434, 0.919863, 0.6567615, 0.67855346, 0.7158917, 0.8198465, 0.7542691, 0.6062059, 0.6686902, 0.63501287, 0.6926691, 0.72867125, 0.6951041, 0.70684266, 0.71531326, 0.6772685, 0.6528353, 0.7362816, 0.63601, 0.77936643, 0.8063371, 0.6890259, 0.85191894, 0.71516806, 0.6893324, 0.7377092, 0.72519547, 0.7886364, 0.77217644, 0.73827344, 0.69471496, 0.5900066]
# loss3 = [0.7478864, 0.5176146, 0.99508494, 0.5526219, 0.84952253, 0.6211705, 0.623885, 0.8659447, 1.4336377, 1.0662161, 0.79201484, 1.0635611, 1.2261451, 0.776972, 1.4339926, 0.54799414, 0.74257994, 0.5809276, 0.58853704, 0.6448275, 1.0649906, 0.50479305, 0.56065726, 0.5593879, 0.6323933, 1.4344658, 0.60004276, 0.93109834, 1.3468184, 0.73881483, 0.63336366, 0.9452125, 0.59955955, 1.3470235, 0.6134988, 0.7207058, 0.5145111, 0.81523407, 1.2332448, 1.0633569, 0.84680396, 1.0641738, 0.5457882, 0.79538774, 1.0658077, 0.878446, 0.74644434, 0.6185902, 0.9436271, 1.3470236, 0.58651674, 0.58268905, 0.8430806, 0.6177658, 0.6168542, 0.8762801, 1.0664203, 0.5524443, 0.81006294, 0.49404827, 0.60443234, 1.3488705, 1.229695, 0.616458, 0.92893404, 0.8144661, 0.92327994, 0.55242366, 0.5639483, 1.0645823, 0.984824, 0.784089, 0.5516486, 1.3476392, 0.9235674, 0.767452, 0.8105336, 0.5058217, 0.7392804, 0.53026134, 1.067033, 0.55398685, 1.4344659, 1.3474339, 0.624675, 0.92502373, 0.7789409, 0.6458742, 0.6079641, 0.618004, 0.9818415, 0.5685634, 0.6119797, 0.49799216, 0.5792066, 0.81094927, 1.0674416, 0.9659645, 1.4347026, 0.5820475]
# loss4 = [0.7153975, 0.70899886, 0.7476115, 0.85065836, 0.9065834, 1.0851445, 0.68965566, 0.6824219, 0.8670215, 0.7142449, 1.0849179, 0.7132421, 0.8683681, 0.54433966, 0.8671799, 0.6435022, 0.7436578, 0.5816668, 0.8681305, 1.6499416, 0.6049035, 0.7304026, 0.60091126, 1.6503325, 0.66201395, 0.66816926, 1.6504627, 0.8680513, 1.0876082, 0.65561754, 0.7437952, 0.6835945, 1.0867304, 1.0846914, 0.5590385, 1.0858241, 0.92017, 0.8299892, 0.67341334, 0.5261126, 1.0862772, 0.83641976, 0.91178864, 0.6745161, 0.5759577, 0.62679297, 0.7181778, 1.0873644, 1.089559, 0.93032706, 0.8398836, 0.8639256, 0.7186137, 0.7338813, 1.0878521, 0.7467043, 0.7046241, 0.86765516, 0.68675303, 1.6503325, 0.8668631, 0.80149484, 0.57790756, 0.7212497, 0.67995477, 0.72092843, 0.58041507, 0.60064316, 0.6843547, 1.0865037, 0.71847534, 0.8671799, 1.650984, 0.68288404, 0.7131495, 0.6591386, 0.8923703, 0.62066525, 0.6371309, 0.86773443, 0.74203515, 0.5481201, 0.8093595, 0.56646144, 0.60306215, 1.0842384, 1.0846915, 0.65045905, 0.5658318, 0.6924332, 0.7517529, 0.8515134, 0.7513092, 0.7000597, 1.0898027, 0.7588981, 0.5851512, 1.0846915, 0.57422817, 0.7267563]
# loss5 = [0.5508401, 0.7270112, 0.58925265, 0.74672544, 0.6288511, 0.7973461, 0.50808287, 0.8046688, 0.5939667, 0.8276794, 0.9939211, 0.9939212, 0.9940913, 0.8040964, 0.6467024, 0.75363845, 0.675952, 0.6627066, 0.55956125, 0.83250666, 0.6967259, 0.5890819, 0.8331497, 0.7745808, 0.694971, 0.7189031, 0.813237, 0.5722742, 0.6058161, 0.6181418, 1.4415828, 0.7805424, 0.83318156, 0.8623192, 0.5304381, 0.78618574, 0.709862, 0.58279055, 0.58718985, 0.5597777, 0.6993647, 0.61772215, 0.6664822, 0.81712943, 0.799141, 0.83395356, 0.993411, 0.5122906, 0.49272466, 0.5830528, 0.64408517, 0.6030815, 0.5932887, 0.8153659, 1.4418693, 0.7189683, 0.690346, 0.75936353, 0.9946012, 0.67593306, 0.5217231, 0.616688, 0.99477136, 0.7496971, 0.6679067, 0.59494674, 0.99358106, 0.72160536, 0.58001655, 0.6368441, 0.7428775, 0.70975095, 0.7418828, 0.5991801, 0.7405779, 0.7449534, 0.60238576, 0.572449, 0.80505025, 0.6504358, 0.83379275, 0.8074105, 0.5934953, 0.9917107, 0.70535034, 0.78332347, 0.57490844, 1.4415828, 0.5501482, 0.73047006, 0.53404164, 0.56777894, 0.7193084, 0.60389775, 0.83247966, 0.585825, 0.724178, 0.662156, 0.7357388, 0.7535118]


line, = ax1.plot(t, accuracy, color='blue', lw=2)
line2, = ax1.plot(t, loss, color='red', lw=2)

plt.grid(True)
plt.show()

