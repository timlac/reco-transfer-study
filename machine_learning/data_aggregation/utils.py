def update_row(item, vals, suffix):
    item.update({f"{col}{suffix}": val for col, val in vals.items()})
    return item
