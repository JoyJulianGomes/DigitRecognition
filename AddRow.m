function AddRow(table,m,l,tc,vc,ta,va)
import mlreportgen.dom.*;
        row = TableRow;
        append(row, TableEntry(sprintf('M0%i_L0%1.2f',m,l)));
        append(row, TableEntry(sprintf('%i',m)));
        append(row, TableEntry(sprintf('%1.2f',l)));
        append(row, TableEntry(sprintf('%2.6f',tc)));
        append(row, TableEntry(sprintf('%2.6f',vc)));
        append(row, TableEntry(sprintf('%2.2f%%',ta)));
        append(row, TableEntry(sprintf('%2.2f%%',va)));
        append(table,row);
end