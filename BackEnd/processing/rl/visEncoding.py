class VisEncoding:
    pos_row = None #slice(up, down, 1) or int
    pos_col = None #slice(left, right, 1) or int
    vis_type = None #['unit visualization', 'bar chart', 'box plot', 'strip plot', 'parallel coordinte plot', 'pie chart', 'line chart', 'horizon graph', 'scatter plot']
    insight_type = None
    insight_value = None
    x = None
    y = None
    size = None
    color = None
    shape = None
    align = None
    is_horizontal = None
    theta = None
    radius = None
    scale = None
    rec_row_type = 'subtree'
    rec_row_priority = 0
    rec_col_type = 'subtree'
    rec_col_priority = 0
    rec_list = None
    def __init__(self, pos_row=None, pos_col=None, vis_type=None, insight_type=None, insight_value=None, x=None, y=None, size=None, color=None, shape=None, align=None, is_horizontal=None\
        , theta=None, radius=None, scale=None, rec_row_type=None, rec_row_priority=None, rec_col_type=None, rec_col_priority=None, rec_list=None):
        self.pos_row = pos_row
        self.pos_col = pos_col
        self.vis_type = vis_type
        self.insight_type = insight_type
        self.insight_value = insight_value
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.shape = shape
        self.align = align
        self.is_horizontal = is_horizontal
        self.theta = theta
        self.radius = radius
        self.scale = scale
        if rec_row_type != None:
            self.rec_row_type = rec_row_type
            self.rec_row_priority = rec_row_priority
            self.rec_col_type = rec_col_type
            self.rec_col_priority = rec_col_priority
            self.rec_list = rec_list
        