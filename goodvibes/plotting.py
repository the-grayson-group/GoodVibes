from . import constants
import matplotlib.path as mpath
import matplotlib.patches as mpatches


# Scatter points that may overlap when graphing
def jitter(datasets, color, ax, nx, marker, edgecol="black"):
    import numpy as np

    for i, p in enumerate(datasets):
        y = [p]
        x = np.random.normal(nx, 0.015, size=len(y))
        ax.plot(
            x,
            y,
            alpha=0.5,
            markersize=7,
            color=color,
            marker=marker,
            markeredgecolor=edgecol,
            markeredgewidth=1,
            linestyle="None",
        )


# Graph a reaction profile
def graph_reaction_profile(graph_data, log, options, plt):
    log.write("\n   Graphing Reaction Profile\n")
    data = {}
    # Get PES data
    for i, path in enumerate(graph_data.path):
        g_data = []
        zero_val = graph_data.qhg_zero[i][0]
        for j, e_abs in enumerate(graph_data.e_abs[i]):
            species = graph_data.qhg_abs[i][j]
            relative = species - zero_val
            if graph_data.units == "kJ/mol":
                formatted_g = constants.J_TO_AU / 1000.0 * relative
            else:
                formatted_g = constants.KCAL_TO_AU * relative  # Defaults to kcal/mol
            g_data.append(formatted_g)
        data[path] = g_data

    # Grab any additional formatting for graph
    with open(options.graph) as f:
        yaml = f.readlines()
    # defaults
    ylim, color, show_conf, show_gconf, show_title = None, None, True, False, True
    label_point, label_xaxis, dpi, dec, legend = (
        False,
        True,
        False,
        2,
        False,
    )
    colors, gridlines, title = None, False, "Potential Energy Surface"
    for i, line in enumerate(yaml):
        if line.strip().find("FORMAT") > -1:
            for j, line in enumerate(yaml[i + 1 :]):
                if line.strip().find("ylim") > -1:
                    try:
                        ylim = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .replace(" ", "")
                            .strip()
                            .split(",")
                        )
                    except IndexError:
                        pass
                if line.strip().find("color") > -1:
                    try:
                        colors = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .replace(" ", "")
                            .strip()
                            .split(",")
                        )
                    except IndexError:
                        pass
                if line.strip().find("title") > -1:
                    try:
                        title_input = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                        )
                        if title_input == "false" or title_input == "False":
                            show_title = False
                        else:
                            title = title_input
                    except IndexError:
                        pass
                if line.strip().find("dec") > -1:
                    try:
                        dec = int(
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                        )
                    except IndexError:
                        pass
                if line.strip().find("pointlabel") > -1:
                    try:
                        label_input = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                            .lower()
                        )
                        if label_input == "false":
                            label_point = False
                    except IndexError:
                        pass
                if line.strip().find("show_conformers") > -1:
                    try:
                        conformers = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                            .lower()
                        )
                        if conformers == "false":
                            show_conf = False
                    except IndexError:
                        pass
                if line.strip().find("show_gconf") > -1:
                    try:
                        gconf_input = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                            .lower()
                        )
                        if gconf_input == "true":
                            show_gconf = True
                    except IndexError:
                        pass
                if line.strip().find("xlabel") > -1:
                    try:
                        label_input = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                            .lower()
                        )
                        if label_input == "false":
                            label_xaxis = False
                    except IndexError:
                        pass
                if line.strip().find("dpi") > -1:
                    try:
                        dpi = int(
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                        )
                    except IndexError:
                        pass
                if line.strip().find("legend") > -1:
                    try:
                        legend_input = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                            .lower()
                        )
                        if legend_input == "false":
                            legend = False
                    except IndexError:
                        pass
                if line.strip().find("gridlines") > -1:
                    try:
                        gridline_input = (
                            line.strip()
                            .replace(":", "=")
                            .split("=")[1]
                            .strip()
                            .split(",")[0]
                            .lower()
                        )
                        if gridline_input == "true":
                            gridlines = True
                    except IndexError:
                        pass
    # Do some graphing
    Path = mpath.Path
    fig, ax = plt.subplots()
    for i, path in enumerate(graph_data.path):
        for j in range(len(data[path]) - 1):
            if colors is not None:
                if len(colors) > 1:
                    color = colors[i]
                else:
                    color = colors[0]
            else:
                color = "k"
                colors = ["k"]
            if j == 0:
                path_patch = mpatches.PathPatch(
                    Path(
                        [
                            (j, data[path][j]),
                            (j + 0.5, data[path][j]),
                            (j + 0.5, data[path][j + 1]),
                            (j + 1, data[path][j + 1]),
                        ],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                    ),
                    label=path,
                    fc="none",
                    transform=ax.transData,
                    color=color,
                )
            else:
                path_patch = mpatches.PathPatch(
                    Path(
                        [
                            (j, data[path][j]),
                            (j + 0.5, data[path][j]),
                            (j + 0.5, data[path][j + 1]),
                            (j + 1, data[path][j + 1]),
                        ],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                    ),
                    fc="none",
                    transform=ax.transData,
                    color=color,
                )
            ax.add_patch(path_patch)
            plt.hlines(data[path][j], j - 0.15, j + 0.15)
        plt.hlines(data[path][-1], len(data[path]) - 1.15, len(data[path]) - 0.85)

    if show_conf:
        markers = ["o", "s", "x", "P", "D"]
        for i in range(len(graph_data.g_qhgvals)):  # i = reaction pathways
            for j in range(len(graph_data.g_qhgvals[i])):  # j = reaction steps
                for k in range(len(graph_data.g_qhgvals[i][j])):  # k = species
                    zero_val = graph_data.g_species_qhgzero[i][j][k]
                    points = graph_data.g_qhgvals[i][j][k]
                    points[:] = [
                        (
                            (x - zero_val)
                            + (graph_data.qhg_abs[i][j] - graph_data.qhg_zero[i][0])
                            + (graph_data.g_rel_val[i][j] - graph_data.qhg_abs[i][j])
                        )
                        * constants.KCAL_TO_AU
                        for x in points
                    ]
                    if len(colors) > 1:
                        jitter(points, colors[i], ax, j, markers[k])
                    else:
                        jitter(points, color, ax, j, markers[k])
                    if show_gconf:
                        plt.hlines(
                            (graph_data.g_rel_val[i][j] - graph_data.qhg_zero[i][0])
                            * constants.KCAL_TO_AU,
                            j - 0.15,
                            j + 0.15,
                            linestyles="dashed",
                        )

    # Annotate points with energy level
    if label_point:
        for i, path in enumerate(graph_data.path):
            for i, point in enumerate(data[path]):
                if dec is 1:
                    ax.annotate(
                        "{:.1f}".format(point),
                        (i, point - fig.get_figheight() * fig.dpi * 0.025),
                        horizontalalignment="center",
                    )
                else:
                    ax.annotate(
                        "{:.2f}".format(point),
                        (i, point - fig.get_figheight() * fig.dpi * 0.025),
                        horizontalalignment="center",
                    )
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if show_title:
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Reaction Profile")
    ax.set_ylabel(r"$G_{rel}$ (kcal / mol)")
    plt.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(which="minor", labelright=True, right=True)
    ax.tick_params(labelright=True, right=True)
    if gridlines:
        ax.yaxis.grid(linestyle="--", linewidth=0.5)
        ax.xaxis.grid(linewidth=0)
    ax_label = []
    xaxis_text = []
    newax_text_list = []
    for i, path in enumerate(graph_data.path):
        newax_text = []
        ax_label.append(path)
        for j, e_abs in enumerate(graph_data.e_abs[i]):
            if i is 0:
                xaxis_text.append(graph_data.species[i][j])
            else:
                newax_text.append(graph_data.species[i][j])
        newax_text_list.append(newax_text)
    # Label rxn steps
    if label_xaxis:
        if colors is not None:
            plt.xticks(range(len(xaxis_text)), xaxis_text, color=colors[0])
        else:
            plt.xticks(range(len(xaxis_text)), xaxis_text, color="k")
        locs, labels = plt.xticks()
        newax = []
        for i in range(len(ax_label)):
            if i > 0:
                y = ax.twiny()
                newax.append(y)
        for i in range(len(newax)):
            newax[i].set_xticks(locs)
            newax[i].set_xlim(ax.get_xlim())
            if len(colors) > 1:
                newax[i].tick_params(axis="x", colors=colors[i + 1])
            else:
                newax[i].tick_params(axis="x", colors="k")
            newax[i].set_xticklabels(newax_text_list[i + 1])
            newax[i].xaxis.set_ticks_position("bottom")
            newax[i].xaxis.set_label_position("bottom")
            newax[i].xaxis.set_ticks_position("none")
            newax[i].spines["bottom"].set_position(("outward", 15 * (i + 1)))
            newax[i].spines["bottom"].set_visible(False)
    else:
        plt.xticks(range(len(xaxis_text)))
        ax.xaxis.set_ticklabels([])
    if legend:
        plt.legend()
    if dpi is not False:
        plt.savefig("Rxn_profile_" + options.graph.split(".")[0] + ".png", dpi=dpi)
    plt.show()
