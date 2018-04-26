import mlreportgen.dom.*;
doc = Document('test');

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
append(row, TableEntry('Epoch'));
append(row, TableEntry('Training Cost'));
append(row, TableEntry('Validation Cost'));
append(row, TableEntry('Training Accuracy'));
append(row, TableEntry('Validation Accuracy'));
append(table,row);


model = 1;
lambda = 1:4;
epoch = 2;
for i=1:size(lambda,2)
    
    for j=1:epoch
        row = TableRow;
        tc=model+i;vc=model+j;ta=tc/vc;va=vc/tc;
        append(row, TableEntry(sprintf('M%i_L%i_E%i',model,lambda(1,i),j)));
        append(row, TableEntry(sprintf('%i',model)));
        append(row, TableEntry(sprintf('%1.2f',lambda(1,i))));
        append(row, TableEntry(sprintf('%i',j)));
        append(row, TableEntry(sprintf('%2.6f',tc)));
        append(row, TableEntry(sprintf('%2.6f',vc)));
        append(row, TableEntry(sprintf('%2.2f%%',ta)));
        append(row, TableEntry(sprintf('%2.2f%%',va)));
        append(table,row);
    end
    
end
append(doc,table);

close(doc);
rptview(doc.OutputPath);