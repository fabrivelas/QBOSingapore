import datetime
import io
from urllib.request import urlopen

import matplotlib.dates as mdates
import numpy as np
from bokeh import palettes
from bokeh.core.templates import get_env
from bokeh.layouts import column
from bokeh.models import AdaptiveTicker, CustomJSTickFormatter, Range1d, RangeTool
from bokeh.plotting import figure, save, show
from bokeh.resources import CDN
from requests import get
from scipy.interpolate import griddata

# ----------------------------------------------------------------------------------------
# controls

# where to expect the Singapore data
DATA_SIN_URL = "https://www.atmohub.kit.edu/data/singapore.dat"

# where to expect the QBO data
DATA_QBO_URL = "https://www.atmohub.kit.edu/data/qbo.dat"

# only show the plot, don't save it?
SHOW = False

# if the specified palette is not available, it will fall back to 'Spectral8'
COLOR_PALETTE = "RdBu"

SIZING_MODE = "stretch_width"

# ----------------------------------------------------------------------------------------

template_html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{ title }}</title>
    {{ bokeh_css }}
    {{ bokeh_js }}
    <script type="text/javascript">
        Bokeh.set_log_level("info");
    </script>
    <style>
      .bk-Column {
        width: 100%;
        height: 750px;
      }
    </style>
  </head>
  <body>
    {{ plot_div }}
    {{ plot_script }}
  </body>
</html>
"""

# ----------------------------------------------------------------------------------------


def get_palette(palette_name: str, n_colors: int = 9):
    """
    Get color palette by name, with n colors.
    Common number of colors are [9, 10, 11, 12, 256].
    If n is not available, interp_palette is used to generate one.
    """

    try:
        palette_full_name = f"{palette_name}{n_colors}"
        return getattr(palettes, palette_full_name)
    except AttributeError:
        pass

    try:
        palette = getattr(palettes, palette_name)
    except AttributeError:
        print(f"Palette {palette_name} not found. Using Spectral8 instead.")
        return palettes.Spectral8

    return palettes.interp_palette(palette[max(palette.keys())], n_colors)


def url_to_file(url, file_name):
    """Dump the content at a URL into a file."""
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


def read_singapore(nmonth, nyear):
    headerlines = []
    ye = []
    dats = []
    count = 0
    date = []

    with urlopen(DATA_SIN_URL) as file:
        for i in range(3):
            headerlines.append(file.readline().decode().strip())
        for year in range(1987, nyear + 1):
            if year == nyear:
                data = np.zeros([nmonth, 15]) * np.nan
            else:
                data = np.zeros([12, 15]) * np.nan
            ye.append(file.readline().strip())
            file.readline()
            if year < 1997:
                data = np.zeros([12, 15]) * np.nan
                for i in range(14):
                    cols = file.readline()
                    cols = cols.strip().split()
                    for j in range(1, 13):
                        if i == 1:
                            date.append(datetime.datetime(year, j, 1))
                        data[j - 1, i] = float(cols[j])
            else:
                for i in range(15):
                    cols = file.readline()
                    cols = cols.strip().split()
                    for j in range(1, 13):
                        if j < nmonth + 1 or year < nyear:
                            if i == 1:
                                date.append(datetime.datetime(year, j, 1))
                            data[j - 1, i] = float(cols[j])
            dats.extend([list(i) for i in data])
            file.readline()
            count = count + 1
    pressure = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 12, 10]
    altitude = -7 * np.log(np.array(pressure) / 1013.25)
    fds = list(mdates.date2num(date))
    fds = np.array(fds)
    return np.array(dats).T[::-1], fds, pressure, altitude


def read_qbo():

    with urlopen(DATA_QBO_URL) as response:
        content = response.read().decode("utf-8")

    data = np.genfromtxt(
        io.StringIO(content),
        skip_header=9,
        dtype=[
            "S6",
            "S4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
            "i4",
        ],
        names=[
            "station",
            "date",
            "p70",
            "n70",
            "p50",
            "n50",
            "p40",
            "n40",
            "p30",
            "n30",
            "p20",
            "n20",
            "p15",
            "n15",
            "p10",
            "n10",
        ],
        delimiter=[6, 4, 6, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2],
        filling_values=-999999,
        missing_values=" ",
    )

    date = []
    dlen = 0
    for i, _ in enumerate(data):
        if data["date"][i]:
            dlen = dlen + 1
            if int(data["date"][i]) > 5000:
                date.append(
                    datetime.datetime.strptime("19" + (data["date"][i]).decode("UTF-8"), "%Y%m")
                )
            else:
                date.append(
                    datetime.datetime.strptime("20" + (data["date"][i]).decode("UTF-8"), "%Y%m")
                )

    fds = mdates.date2num(date)

    up = np.array(
        list(
            [
                data["p70"],
                data["p50"],
                data["p40"],
                data["p30"],
                data["p20"],
                data["p15"],
                data["p10"],
            ]
        )
    )

    pressure = [70, 50, 40, 30, 20, 15, 10]
    altitude = -7 * np.log(np.array(pressure) / 1013.25)

    return up, fds, pressure, altitude


def make_layout(p, select):
    """Stack the two figures vertically, stretching the layout to the container width.

    The root Column must use the same sizing mode as the figures: with a fixed
    Column, the stretch_width children report ~0 required width and the whole
    layout collapses to a sliver.
    """

    return column(p, select, sizing_mode=SIZING_MODE)


def main():
    print("make QBO plot...")

    up1, fds1, pressure1, altitude1 = read_qbo()
    tnmoth = np.shape(up1)[1]  # total number of months
    up = np.zeros([15, tnmoth]) * np.nan
    nmonth = np.shape(up1)[1] % 12  # number of months above a whole year
    nyear = mdates.num2date(fds1[-1]).year
    if nmonth == 0:
        nyear += 1
    # print(nmonth, nyear)

    up2, fds2, pressure, altitude = read_singapore(nmonth, nyear)
    fds = fds1
    # print(fds[-fds2.size], fds[-fds2.size + 1])
    # print(fds2[0])
    # print(np.size(fds1), np.size(fds2))

    fds[-fds2.size :] = fds2
    fds = np.array(fds)
    # print(np.size(fds), fds[-1], fds1[-1], fds2[-1])
    # print(mdates.num2date(fds2[-1]))

    up1 = up1 * 1.0
    up1[up1 < -10000] = np.nan
    up[3, :] = up1[0, :]
    up[5, :] = up1[1, :]
    up[7, :] = up1[2, :]
    up[9, :] = up1[3, :]
    up[11, :] = up1[4,]
    up[12, :] = up1[5, :]
    up[14, :] = up1[6, :]
    up[-up2.shape[0] :, -up2.shape[1] :] = up2
    ind = ~np.isnan(up)
    time, press = np.meshgrid(fds, pressure)
    uu = np.array(up[ind]) * 0.1
    time = time[ind]
    press = press[ind]

    points1 = np.array([[time[i], press[i]] for i in range(len(time))])
    X, Y = np.meshgrid(fds, pressure)
    u = griddata(points1, uu, (X, Y), method="linear")

    deltat = int((np.nanmax(fds) - fds[0]) / 1) + 1

    by0 = fds[0]
    by1 = fds[0] + deltat

    axt = fds[np.nanargmin(abs(fds - by0)) : 1 + np.nanargmin(abs(fds - by1))]

    axz = u[:, np.nanargmin(abs(fds - by0)) : 1 + np.nanargmin(abs(fds - by1))]

    print(
        "data range:\n  from",
        mdates.num2date(axt[0]).strftime("%Y-%m-%d"),
        "to",
        mdates.num2date(axt[-1]).strftime("%Y-%m-%d"),
    )

    p = figure(
        width=1200,
        height=550,
        y_axis_type="log",
        y_range=(max(pressure), min(pressure)),  # type: ignore
        x_range=(  # type: ignore
            mdates.date2num(np.datetime64("1952-03-01")),
            mdates.date2num(np.datetime64("1962-03-01")),
        ),
        y_axis_label="Pressure [hPa]",
        x_axis_type="datetime",
        sizing_mode=SIZING_MODE,
        match_aspect=False,
        toolbar_location="above",
        output_backend="webgl",
    )

    # p.title="Quasi-Biennial-Oscillation (QBO)"
    # p.title.text_font_size = "25px"  # type: ignore
    # p.title.align = "center"  # type: ignore

    contour_levels = list(range(-40, 45, 5))

    contour_renderer = p.contour(
        axt,
        pressure,
        axz,
        contour_levels,
        fill_color=get_palette(COLOR_PALETTE, len(contour_levels)),
        line_color="black",
    )

    select = figure(
        width=1200,
        height=200,
        y_axis_type="log",
        y_range=p.y_range,
        x_axis_label="Time [year]",
        sizing_mode=SIZING_MODE,
        match_aspect=False,
        x_range=(  # type: ignore
            mdates.date2num(np.datetime64("1950-01-01")),
            mdates.date2num(np.datetime64("2030-01-01")),
        ),
        tools="",
        toolbar_location=None,
        output_backend="webgl",
    )

    select.contour(
        axt,
        pressure,
        axz,
        contour_levels,
        line_alpha=0.1,
        fill_color=get_palette(COLOR_PALETTE, len(contour_levels)),
        line_color="black",
    )

    range_tool = RangeTool(x_range=p.x_range, start_gesture="pan")
    range_tool.overlay.fill_color = "red"
    range_tool.overlay.fill_alpha = 0.4
    select.add_tools(range_tool)

    select.title = "Move/resize slider by dragging"
    select.title.text_font_size = "15px"  # type: ignore
    select.axis.axis_label_text_font_size = "20px"  # type: ignore
    select.axis.major_label_text_font_size = "15px"  # type: ignore
    select.yaxis.major_label_text_font_size = "0px"  # type: ignore

    formatter_js_my = """
const date = new Date(tick * 24 * 60 * 60 * 1000);
const month = (date.getUTCMonth() + 1).toString().padStart(2, '0');
const year = date.getUTCFullYear();
return `${month}/${year}`;
    """
    formatter_js_y = """
const date = new Date(tick * 24 * 60 * 60 * 1000);
return `${date.getUTCFullYear()}`;
    """
    formatter_js_iso = """
const date = new Date(tick * 24 * 60 * 60 * 1000);
return `${date.toISOString()}`;
    """

    select.xaxis.ticker = AdaptiveTicker(  # type: ignore
        base=365.25 * 10,  # try to make a tick every 10 years
    )
    select.xaxis[0].formatter = CustomJSTickFormatter(code=formatter_js_y)
    select.yaxis.major_label_text_font_size = "0pt"  # type: ignore
    select.yaxis.major_tick_line_color = None  # type: ignore
    select.yaxis.minor_tick_line_color = None  # type: ignore

    altitude = -7 * np.log(np.array(pressure) / 1013.25)
    p.extra_y_ranges["altitude"] = Range1d(min(altitude), max(altitude))
    p.axis.axis_label_text_font_size = "20px"  # type: ignore
    p.axis.major_label_text_font_size = "15px"  # type: ignore
    p.xaxis.ticker = AdaptiveTicker(  # type: ignore
        base=365.25,  # try to make a tick every year
        min_interval=1,  # max. resolution is 1 day
        max_interval=365 * 10,  # min. resolution is 10 years
    )
    p.xaxis[0].formatter = CustomJSTickFormatter(code=formatter_js_my)

    colorbar = contour_renderer.construct_color_bar()
    colorbar.title = "East to West (-/+) monthly mean zonal winds [m/s]"
    colorbar.title_text_font_size = "15px"
    p.add_layout(colorbar, "below")

    layout = make_layout(p, select)

    if SHOW:
        show(layout)
        return

    save(
        layout,
        filename="plot.html",
        title="QBO-Plot",
        resources=CDN,
        template=get_env().from_string(template_html),
    )


if __name__ == "__main__":
    main()
