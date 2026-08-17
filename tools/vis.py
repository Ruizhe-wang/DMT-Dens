import numpy as np
import umap
import plotly.graph_objs as go
import plotly.io as pio
import argparse

c_list_1 = ["#7eccb4","#7ccbb3","#7bcbb2","#79cab1","#78c9b0","#76c9ae","#75c8ad","#73c7ac","#72c7ab","#70c6aa","#6fc5a8","#6dc5a7","#6cc4a6","#6ac3a4","#69c3a3","#68c2a2","#66c1a1","#65c19f","#64c09e","#62bf9d","#61bf9b","#60be9a","#5ebd98","#5dbd97","#5cbc96","#5bbb94","#59bb93","#58ba92","#57b990","#56b98f","#54b88d","#53b78c","#52b78b","#51b689","#50b588","#4fb486","#4db485","#4cb383","#4bb282","#4ab180","#49b17f","#48b07d","#47af7c","#46ae7b","#45ae79","#44ad78","#43ac76","#42ab75","#40aa73","#3fa972","#3ea870","#3da76f","#3ca66d","#3ba56c","#3aa56a","#39a469","#38a367","#37a266","#36a164","#35a063","#349f61","#339e60","#329d5e","#319c5d","#309b5c","#2f9a5a","#2e9959","#2d9757","#2c9656","#2b9555","#2a9453","#299352","#289251","#27914f","#26904e","#258f4d","#248e4c","#238d4a","#228c49","#218b48","#208a47","#1f8946","#1e8845","#1d8744","#1c8643","#1b8542","#1a8441","#198340","#18823f","#17813e","#16803d","#157f3c","#147e3b","#137d3a","#127c39","#117b38","#107a38","#107937","#0f7836","#0e7735","#0d7634","#0c7534","#0b7433","#0b7332","#0a7232","#097131","#087030","#086f2f","#076e2f","#066c2e","#066b2d","#056a2d","#05692c","#04682b","#04672b","#04662a","#03642a","#036329","#026228","#026128","#026027","#025e27","#015d26","#015c25","#015b25","#015a24","#015824","#015723","#005623","#005522","#005321","#005221","#005120","#005020","#004e1f","#004d1f","#004c1e","#004a1e","#00491d","#00481d","#00471c","#00451c","#00441b"]
c_list_2 = ["#fca26d","#fca06c","#fc9f6b","#fc9e6a","#fc9c68","#fc9b67","#fb9a66","#fb9865","#fb9764","#fb9563","#fb9462","#fb9361","#fb9160","#fa905f","#fa8f5e","#fa8d5d","#fa8c5c","#f98b5b","#f9895a","#f98859","#f98759","#f88558","#f88457","#f88356","#f78155","#f78055","#f77f54","#f67d53","#f67c52","#f67b52","#f57951","#f57850","#f4774f","#f4754f","#f4744e","#f3734d","#f3714c","#f2704c","#f26f4b","#f16d4a","#f16c49","#f06b49","#f06948","#ef6847","#ef6646","#ee6545","#ed6344","#ed6243","#ec6042","#ec5f42","#eb5d41","#ea5c40","#ea5a3f","#e9593e","#e8573c","#e8563b","#e7543a","#e65339","#e65138","#e55037","#e44e36","#e44c35","#e34b34","#e24932","#e14831","#e04630","#e0442f","#df432e","#de412d","#dd402b","#dc3e2a","#dc3c29","#db3b28","#da3927","#d93826","#d83624","#d73423","#d63322","#d53121","#d43020","#d32e1f","#d22c1e","#d12b1d","#d0291b","#cf281a","#ce2619","#cd2518","#cc2317","#cb2216","#ca2015","#c91f14","#c81d13","#c71c12","#c61b11","#c51911","#c31810","#c2170f","#c1150e","#c0140d","#bf130c","#be120c","#bc110b","#bb100a","#ba0e09","#b80d09","#b70c08","#b60b07","#b50b07","#b30a06","#b20906","#b10805","#af0705","#ae0704","#ac0604","#ab0504","#a90503","#a80403","#a60402","#a50302","#a40302","#a20302","#a00201","#9f0201","#9d0201","#9c0101","#9a0101","#990101","#970101","#960100","#940100","#920000","#910000","#8f0000","#8e0000","#8c0000","#8a0000","#890000","#870000","#860000","#840000","#820000","#810000","#7f0000"]
c_list_3 = ["#8fd3be","#8ed3be","#8dd2be","#8bd2bf","#8ad1bf","#88d1c0","#87d0c0","#86cfc1","#84cfc1","#83cec1","#81cec2","#80cdc2","#7fccc3","#7dccc3","#7ccbc4","#7acac4","#79cac5","#77c9c5","#76c8c6","#75c8c6","#73c7c7","#72c6c7","#70c5c7","#6fc5c8","#6ec4c8","#6cc3c9","#6bc3c9","#69c2ca","#68c1ca","#67c0ca","#65bfcb","#64bfcb","#63becb","#61bdcc","#60bccc","#5fbbcc","#5dbacc","#5cb9cc","#5ab9cd","#59b8cd","#58b7cd","#57b6cd","#55b5cd","#54b4cd","#53b3cd","#51b2cd","#50b1cd","#4fb0cd","#4eafcd","#4caecd","#4badcc","#4aaccc","#49abcc","#48aacc","#46a9cb","#45a8cb","#44a6cb","#43a5ca","#42a4ca","#41a3c9","#3fa2c9","#3ea1c8","#3da0c8","#3c9ec7","#3b9dc7","#3a9cc6","#399bc6","#379ac5","#3699c5","#3597c4","#3496c4","#3395c3","#3294c2","#3193c2","#3092c1","#2f90c0","#2d8fc0","#2c8ebf","#2b8dbf","#2a8cbe","#298abd","#2889bd","#2788bc","#2687bc","#2586bb","#2485ba","#2383ba","#2282b9","#2081b9","#1f80b8","#1e7fb7","#1d7eb7","#1c7db6","#1b7bb5","#1a7ab5","#1979b4","#1978b3","#1877b3","#1776b2","#1674b1","#1573b0","#1472b0","#1371af","#1370ae","#126fad","#116dac","#106cac","#106bab","#0f6aaa","#0e69a9","#0e67a8","#0d66a7","#0d65a6","#0c64a5","#0c63a4","#0c61a3","#0b60a2","#0b5fa1","#0a5ea0","#0a5d9e","#0a5b9d","#0a5a9c","#09599b","#09589a","#095699","#095597","#095496","#095395","#085294","#085092","#084f91","#084e90","#084d8e","#084b8d","#084a8c","#08498a","#084889","#084688","#084586","#084485","#084384","#084182","#084081"]


def plot(num_step=400, n_neighbors=50):
    data_meta_0 = np.load('gen_save/gathered_reconstruct_history_history3.npy')
    data_meta_1 = np.load('gen_save/gathered_reconstruct_history_history4.npy')
    data_meta = np.concatenate([data_meta_0, data_meta_1], axis=0)
    data_meta = data_meta[:, 1:, :]

    label_meta_0 = np.load('gen_save/gathered_reconstruct_label3.npy')
    label_meta_1 = np.load('gen_save/gathered_reconstruct_label4.npy')
    label_meta = np.concatenate([label_meta_0, label_meta_1], axis=0)

    data_meta = data_meta[label_meta < 3]
    label_meta = label_meta[label_meta < 3]
    label_meta = label_meta.reshape(-1,1).repeat(num_step, axis=1).T.reshape(-1)

    data_input = []
    label_input = []
    for i in range(num_step):
        data_input.append(data_meta[:, (1000 // num_step) * i, :])
        label_input.append([i] * data_meta.shape[0])

    data_input_np = np.concatenate(data_input, axis=0)
    label_input_np = np.array(label_input).reshape(-1)

    umap_dr = umap.UMAP(n_components=2, n_neighbors=n_neighbors).fit_transform(data_input_np)


    color_list = []
    for i in range(label_input_np.shape[0]):
        # import pdb; pdb.set_trace()
        if label_meta[i] == 0:
            color_list.append(c_list_1[len(c_list_1)*label_input_np[i]//num_step])
        elif label_meta[i] == 1:
            color_list.append(c_list_2[len(c_list_2)*label_input_np[i]//num_step])
        else:
            color_list.append(c_list_3[len(c_list_3)*label_input_np[i]//num_step])

    trace = go.Scatter(
        x=umap_dr[:, 0],
        y=umap_dr[:, 1],
        mode='markers',
        marker=dict(
            size=2,
            color=color_list,
            showscale=False,
        )
    )

    layout = dict(
        # title='UMAP Visualization',
        yaxis=dict(zeroline=False, showgrid=False, visible=False),  # Remove y-axis
        xaxis=dict(zeroline=False, showgrid=False, visible=False),  # Remove x-axis
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=False,
        width=600,
        height=600
    )

    fig = dict(data=[trace], layout=layout)

    pio.write_image(fig, f'umap_visualization_{num_step}_{n_neighbors}.png', format='png', scale=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_step', type=int, default=400)
    parser.add_argument('--n_neighbors', type=int, default=50)
    args = parser.parse_args()

    plot(num_step=args.num_step, n_neighbors=args.n_neighbors)