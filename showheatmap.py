import numpy as np
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.sampledata.unemployment1948 import data
import pandas as pd 
import math 
from bokeh.plotting import figure, output_file, show, VBox


x = pd.read_csv("./docsize_vs_accuracy/svm_cv_count_doclength_accuracy.csv")
x["posnegsum"] = x.poscount + x.negcount
x["posnegdif"] = x.poscount - x.negcount
x["error"] = x.predicted != x.trueclass


# minmm = abs(x.posnegdif_count.min())+1
# x["posnegdif"] = x.posnegdif_count.apply(lambda l : round(math.log(l+minmm,2)))
x["wordcount_rank"] = x.wordcount.apply(lambda l : round(math.log(l,2)))


x["length"] = x["wordcount_rank"]
x["subjectivity"] = x["posnegdif"]

groups = x.groupby(["length","subjectivity"], as_index=False).agg({"error": lambda a : float(sum(a))/len(a)})





# Read in the data with pandas. Convert the year column to string

x = groups

# adjust error range to 0 > 10 
emin = x.error.min()
emax = x.error.max()
oldrange = emax - emin
newrange = 12

x["error"] = x.error.map(lambda e : math.log(e+1))
x["errorrange"] = x.error.map(lambda e : ((e-emin)*newrange / oldrange ))

# this is the colormap from the original plot
colors = ["#F7E9E9","#F7D7D7","#F7BCBC","#F7A3A3","#ED7E7E","#ED6464","#DE4545","#CC3333","#B82727","#9C1616","#800E0E","#630707","#450000"]


# Set up the data for plotting. We will need to have values for every
# pair of year/month names. Map the rate to a color.
length = []
subjective = []
color = []
errorrate = []

for l in x.length:
    for s in x.subjectivity:
        length.append(l)
        subjective.append(s)
        error = x["errorrange"][x.length == l][x.subjectivity == s].values        
        if len(error) > 0 :
            errorrate.append(error[0])        
            color.append(colors[int(error[0])])
        else :
            errorrate.append(None)
            color.append("#FFFFFF")

# EXERCISE: create a `ColumnDataSource` with columns: month, year, color, rate
source = ColumnDataSource(
    data=dict(
        length=length,
        subjective=subjective,
        color=color,
        errorrate=errorrate,
    )
)

# EXERCISE: output to static HTML file

# create a new figure
p = figure(title="doc length vs doc subjective vs error rate", tools="resize",
           x_range=[0,int(max(length))+1], y_range=[int(max(subjective)+1),int(min(subjective))],
           plot_width=400, plot_height=400, x_axis_location="above")


p.rect('length', 'subjective', 0.95, 0.95, source=source, color='color', line_color=None)

# EXERCISE: use the `rect renderer with the following attributes:
#   - x_range is years, y_range is months (reversed)
#   - fill color for the rectangles is the 'color' field
#   - line_color for the rectangles is None
#   - tools are resize and hover tools
#   - add a nice title, and set the plot_width and plot_height
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi/3



output_file("showheatmap.html")
show(VBox(p))
