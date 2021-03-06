#!/bin/lua

function parsefile(filename)
    minttyfilename = filename .. ".minttyrc"
	file = io.open(filename .. ".tex", "w")
	count = 0
	for line in io.lines(minttyfilename) do
		count = count + 1
    end
    if count > 17 then
        -- Assume a well formatted mintty export
        for line in io.lines(minttyfilename) do
            for color, value in string.gmatch(line, "(.+)=(.+)") do
                file:write("\\definecolor{")
                file:write(color)
                file:write("}{RGB}{")
                file:write(value)
                file:write("}\n")
            end
		end
    else
        -- Assume empty file. Mix colors.
		file:write("\\colorlet{BackgroundColour}{white}\n")
		file:write("\\colorlet{ForegroundColour}{black}\n")
		file:write("\\colorlet{CursorColour}{black}\n")
		file:write("\\colorlet{Black}{black}\n")
		file:write("\\colorlet{BoldBlack}{black!80!white}\n")
		file:write("\\colorlet{Red}{red}\n")
		file:write("\\colorlet{BoldRed}{red!80!white}\n")
		file:write("\\colorlet{Green}{green}\n")
		file:write("\\colorlet{BoldGreen}{lime}\n")
		file:write("\\colorlet{Yellow}{yellow}\n")
		file:write("\\colorlet{BoldYellow}{orange}\n")
		file:write("\\colorlet{Blue}{blue}\n")
		file:write("\\colorlet{BoldBlue}{blue!80!white}\n")
		file:write("\\colorlet{Magenta}{magenta}\n")
		file:write("\\colorlet{BoldMagenta}{violet}\n")
		file:write("\\colorlet{Cyan}{cyan}\n")
		file:write("\\colorlet{BoldCyan}{teal}\n")
		file:write("\\colorlet{White}{darkgray}\n")
		file:write("\\colorlet{BoldWhite}{gray}\n")
	end
	file:close()
end

parsefile("terminalcolors")