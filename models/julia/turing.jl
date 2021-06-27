using CSV, DataFrames, Turing, ArviZ

pl_data = DataFrame(CSV.File("data/premierleague.csv"))