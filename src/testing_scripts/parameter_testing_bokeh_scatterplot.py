from bokeh.io import curdoc, show, output_notebook
from bokeh.layouts import layout
from bokeh.models import HoverTool, ColumnDataSource, Range1d, TableColumn, DataTable, StringFormatter
from bokeh.plotting import figure
import pandas as pd
import numpy as np

output_notebook()

def create_scatterplot(df, x_axis, y_axis, title, source):
    p = figure(
        title=title,
        width=650,
        height=650,
        x_axis_label=x_axis,
        y_axis_label=y_axis,
        tools=["lasso_select", "pan", "wheel_zoom", "reset", "box_select", "tap"]
    )

    p.scatter(
        x=x_axis,
        y=y_axis,
        size=5,
        color="firebrick",
        alpha=0.5,
        source=source,
        selection_color="deepskyblue",
        nonselection_alpha=0.2
    )
    
    # —————————————————————————————
    #  font-size tweaks:
    # —————————————————————————————
    p.title.text_font_size           = "18pt"
    p.title.align                     = "center"
    
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    # —————————————————————————————

    # Adjust axis ranges
    x_axis_offset = np.max(df[x_axis]) * 0.1
    y_axis_offset = np.max(df[y_axis]) * 0.1

    p.x_range = Range1d(min(df[x_axis]) - x_axis_offset, 1)
    p.y_range = Range1d(min(df[y_axis]) - y_axis_offset, max(df[y_axis]) + y_axis_offset)

    # Limit the amount users can pan
    p.x_range.bounds = (0, 1)
    p.y_range.bounds = (
        min(df[y_axis]) - y_axis_offset*2,
        max(df[y_axis]) + y_axis_offset*2
    )

    return p

# load
parameter_search_results = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures/mm10/filtered_L2_E7.5_rep1/parameter_search/parameter_search/grid_search_results.csv"
df = pd.read_csv(parameter_search_results)

# rename to human labels
df = df.rename(columns={
    'colsample_bytree' : 'Columns per Tree',
    'gamma'            : 'Loss Split Requirement',
    'reg_alpha'        : 'L1 Reg',
    'reg_lambda'       : 'L2 Reg',
    'subsample'        : 'Frac of Training Data',
    'n_estimators'     : 'Trees',
    'max_depth'        : 'Max Depth',
    'learning_rate'    : 'Learning Rate',
    'val_ap'           : 'Precision',
    'val_auc'          : 'AUC',
    'imp_entropy'      : 'Feature Importance Entropy',
    'imp_cv'           : 'Feature Importance Variation'
})

source = ColumnDataSource(data=df)

# use new names here:
plot = create_scatterplot(
    df,
    x_axis="AUC", 
    y_axis="Feature Importance Entropy", 
    title="AUC vs Feature Importance Entropy", 
    source=source
)

columns = [
    TableColumn(
        field='Columns per Tree',
        title='% Columns per Tree',
        formatter=StringFormatter(text_color='black')
    ),
    TableColumn(
        field='Loss Split Requirement',
        title='% Loss Split Requirement',
        formatter=StringFormatter(text_color='black')
    ),
    TableColumn(
        field='Learning Rate',
        title='Learning Rate',
        formatter=StringFormatter(text_color='black')
    ),
    TableColumn(
        field='Trees',
        title='Trees',
        formatter=StringFormatter(text_color='black')
    ),
    TableColumn(
        field='L1 Reg',
        title='L1 Reg',
        formatter=StringFormatter(text_color='black')
    ),
    TableColumn(
        field='Frac of Training Data',
        title='% Frac of Training Data',
        formatter=StringFormatter(text_color='black')
    ),
]

css = """
.slick-cell {
    text-align: center;
    font-size: 14px !important;
}

.slick-header-column {
    text-align: center;
    height: 40px !important;
    font-size: 14px !important;
    white-space: normal !important;
    line-height: 1.2em;
}
"""

data_table = DataTable(
    source=source,
    columns=columns,
    height=650,
    width=650,
    selectable=True
)
data_table.stylesheets = [css]

map_layout = layout([[plot, data_table]])
show(map_layout)
