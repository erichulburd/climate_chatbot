let fs = require('fs');
let csv = require('csv-parse');
let _ = require('lodash');

let raw = fs.readFileSync('./data/climate_augmented_data.csv', 'utf8');

let form = ``;
let questionTemplate = _.template(`
<fieldset>
  <label style="font-weight: bold; font-size: 0.9em;"><%= index + 1 %>. <%= question %></label>
  <p id="bdab0417-6da6-412e-a4a9-2bbde53ecc6f-<%= index %>" style="font-style:italic;font-size:0.75em;"><%= answer %></p>
  <textarea class="form-control" cols="120" name="question-<%= index %>" rows="5"></textarea>
</fieldset>
`);

csv(raw, (err, out) => {
  if (err) process.exit(err);
  console.info(out.length);
  let i = 0;
  out.forEach((row) => {
    let question = row[0];
    let answer = row[1];
    form += questionTemplate({ question: question, answer: answer, index: i});
    i++;
  });
  fs.writeFileSync('./working_dir/form.html', form, 'utf8');
});
