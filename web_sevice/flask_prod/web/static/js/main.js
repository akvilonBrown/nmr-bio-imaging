$(document).ready(function () {
  console.log("external");
  $("form input").change(function () {
    $("form p").text(this.files.length + " file(s) selected");
  });
});
