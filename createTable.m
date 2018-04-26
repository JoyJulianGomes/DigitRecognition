function table = createTable()
    import mlreportgen.dom.*;
    
    table = Table(8);
    table.Border = 'single';
    table.ColSep = 'single';
    table.RowSep = 'single';
    table.BackgroundColor='#fff2c6';
    table.HAlign='center';
    table.TableEntriesHAlign='center';

    row = TableRow;%TableHeader
    append(row, TableEntry('Config-ID'));
    append(row, TableEntry('Model'));
    append(row, TableEntry('Lambda'));
    append(row, TableEntry('Training Cost'));
    append(row, TableEntry('Validation Cost'));
    append(row, TableEntry('Training Accuracy'));
    append(row, TableEntry('Validation Accuracy'));
    append(table,row);
end