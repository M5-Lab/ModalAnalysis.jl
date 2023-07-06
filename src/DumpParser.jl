struct LammpsDump{HD,DD}
    header_length::UInt32
    header_data::HD
    n_lines::UInt32
    n_samples::UInt32
    data_storage::DD
    path::String
end


function LammpsDump(path; header_length = 9)
    header_data, n_lines = parse_dump_header(path, header_length)
    n_samples = UInt32(n_lines / (header_data["N_atoms"] + header_length))
    # Build dataframe to hold data -- reuse memory
    names = tuple(Symbol.(header_data["fields"])...)
    values = (zeros(header_data["N_atoms"]) for _ in 1:length(header_data["fields"]))
    dump_data = DataFrame( NamedTuple{names}(values));

    return LammpsDump{typeof(header_data), typeof(dump_data)}(header_length, header_data, n_lines, n_samples, dump_data, path)
end

Base.length(ld::LammpsDump) = ld.n_samples
Base.iterate(ld::LammpsDump, state = 1) = state > length(ld) ? nothing : (parse_timestep!(ld, state), state + 1)
haskey(ld::LammpsDump, x::String) = x ∈ names(ld.data_storage)
get_col(ld::LammpsDump, x::String) = getproperty(ld.data_storage, Symbol(x))

function skiplines(path::String, num_lines)
    io = open(path, "r")
    for _ in range(1, num_lines)
        skipchars(!=('\n'), io)
        read(io, Char)
    end
    return io
end

function skiplines(io::IOStream, num_lines)
    for _ in range(1, num_lines)
        skipchars(!=('\n'), io)
        read(io, Char)
    end
    return io
end

function parse_dump_header(dump_path, dump_header_len)
    file = open(dump_path, "r")
    n_lines = countlines(file) #need to close file after getting this data
    close(file)

    file = open(dump_path, "r")

    #parse first header for useful data
    header_data = Dict("N_atoms" => 0.0, "L_x" => [0.0,0.0], "L_y" => [0.0,0.0],
         "L_z" => [0.0,0.0], "fields" => nothing)
    for i in range(1, dump_header_len)
        line = readline(file)
        if i == 4
            header_data["N_atoms"] = parse(Int, line)
        elseif i == 6
            header_data["L_x"]  = [parse(Float64,x) for x in split(strip(line))]
        elseif i == 7
            header_data["L_y"] = [parse(Float64,y) for y in split(strip(line))]
        elseif i == 8
            header_data["L_z"] = [parse(Float64,z) for z in split(strip(line))]
        elseif i == 9
            header_data["fields"] = split(strip(line))[3:end]
        end
    end

    return header_data, n_lines
end

function parse_timestep!(ld::LammpsDump, sample_number)

    lines_to_skip = (sample_number - 1)*(ld.header_length + ld.header_data["N_atoms"]) + (ld.header_length)
    io = skiplines(ld.path, lines_to_skip)
    #Parse atom data
    for j in range(1, ld.header_data["N_atoms"])
        ld.data_storage[j,:] .= parse.(Float64, split(strip(readline(io))))
    end

    return ld
end