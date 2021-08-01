using DataFrames

"""
    score_table(data)
Convert Home, score1, score2, Away -> final table
"""
function score_table(data)
    data[!, :HomePoints] = ifelse.(
        data[!, :score1] .> data[!, :score2], 
        3, 
        ifelse.(
            data[!, :score1] .== data[!, :score2],
            1, 
            0
        )
    )
    data[!, :AwayPoints] = ifelse.(
        data[!, :score2] .> data[!, :score1], 
        3, 
        ifelse.(
            data[!, :score2] .== data[!, :score1],
            1, 
            0
        )
    )

    data[!, :HomeGD] = data[!, :score1] - data[!, :score2]
    data[!, :AwayGD] = data[!, :score2] - data[!, :score1]

    home = groupby(data, :Home)
    home = combine(home, :HomePoints => sum => :HomePoints, :HomeGD => sum => :HomeGD)
    rename!(home, :Home => :Team)

    away = groupby(data, :Away)
    away = combine(away, :AwayPoints => sum => :AwayPoints, :AwayGD => sum => :AwayGD)
    rename!(away, :Away => :Team)

    data = leftjoin(home, away, on=:Team)

    data[!, :Points] = data[!, :HomePoints] + data[!, :AwayPoints]
    data[!, :GD]     = data[!, :HomeGD] - data[!, :AwayGD]

    select!(data, [:Team, :Points, :GD])
    sort!(data, [order(:Points, rev=true), order(:GD, rev=true)])

    return data
end
